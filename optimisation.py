from dataclasses import dataclass, field
import os
import sys
import argparse
import logging
import time

import torch

import hktex
from hktex.data import MeshSamplerDataModule
from hktex.trainers import BaseTrainer
from hktex.utils.video import save_video, combine_videos
from hktex.utils.typing import *

from hktex.utils import repr_patches

__all__ = ["repr_patches", "show_video"]


class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


def main(args, extras, render=True) -> Dict[str, Any]:
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger = logging.getLogger("hktex")
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
            handler.addFilter(ColoredFilter())

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    from hktex.utils import (
        ExperimentConfig,
        load_config,
        seed_everything,
        save_config_snapshot,
    )

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(
        args.config, args.rendering_config, cli_args=extras, n_gpus=n_gpus
    )

    seed_everything(cfg.seed)

    datamodule: MeshSamplerDataModule = hktex.find(cfg.data_type)(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")

    trainer: BaseTrainer = hktex.find(cfg.trainer_type)(
        cfg.trainer, datamodule, renderer_cfg=cfg.renderer
    )

    # Add output logs
    if cfg.optim.save_logs:
        fh = logging.FileHandler(os.path.join(cfg.trial_dir, "logs.txt"))
        fh.setLevel(logging.INFO)
        if args.verbose:
            fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    # Save raw and parsed config
    save_config_snapshot(os.path.join(cfg.trial_dir, "configs"), cfg, args.config)

    # Save args and extras
    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    write_to_text(
        os.path.join(cfg.trial_dir, "cmd.txt"),
        ["python " + " ".join(sys.argv), str(args), str(extras)],
    )

    debug_log_dir = None
    if hktex.is_debug():
        debug_log_dir = os.path.join(cfg.trial_dir, "debug_logs")
        os.makedirs(debug_log_dir, exist_ok=True)

    optimise_start = time.time()
    v_colours, gt_colours, init_colours = trainer.optimise(
        n_iter=cfg.optim.iters, debug_log_dir=debug_log_dir
    )
    optimise_end = time.time()
    hktex.info(f"Optimise took {optimise_end-optimise_start:.2f} seconds")

    if not render:
        return {
            "optimisation": trainer,
            "datamodule": datamodule,
            "colours": (v_colours, gt_colours, init_colours),
        }

    torch.cuda.empty_cache()
    gt_renderings = trainer.render_gt(cfg.renderer.n_rotating_frames)
    render_start = time.time()
    result_renderings = trainer.render_result(cfg.renderer.n_rotating_frames)
    render_end = time.time()
    hktex.info(f"Rendering results took {render_end-render_start:.2f} seconds")
    if hasattr(trainer, "render_kernel_rings"):
        ring_renderings = trainer.render_kernel_rings(cfg.renderer.n_rotating_frames)
    else:
        ring_renderings = None
    if cfg.renderer.n_rotating_frames > 1:
        if ring_renderings is not None:
            combined_renderings = combine_videos(
                gt_renderings, result_renderings, ring_renderings
            )
        else:
            combined_renderings = combine_videos(gt_renderings, result_renderings)
    else:
        combined_renderings = None

    if cfg.optim.save_model:
        save_dir = os.path.join(cfg.trial_dir, "ckpts")
        os.makedirs(save_dir, exist_ok=True)
        torch_size, npz_size = trainer.save_model(
            os.path.join(save_dir, cfg.optim.save_model_name)
        )
        if torch_size is not None:
            hktex.info(f"Pytorch model saved: {torch_size:.2f} KB")
        if npz_size is not None:
            hktex.info(f"Numpy model saved: {npz_size:.2f} KB")
        save_video(combined_renderings, os.path.join(save_dir, "gt_vs_out.mp4"))

    return {
        "optimisation": trainer,
        "datamodule": datamodule,
        "colours": (v_colours, gt_colours, init_colours),
        "renderings": (
            gt_renderings,
            result_renderings,
            ring_renderings,
            combined_renderings,
        ),
        "storage": (torch_size, npz_size),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--rendering_config",
        default="configs/rendering.yaml",
        help="path to rendering config file",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `train.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    args, extras = parser.parse_known_args()
    main(args, extras)
