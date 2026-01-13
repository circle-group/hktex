import sys
from pathlib import Path
import os


sys.path.append(str(Path(__file__).resolve().parent.parent))

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

import torch
from functools import partial

import heatsplats
from heatsplats.utils import (
    load_config,
    ExperimentConfig,
)
from heatsplats.data import MeshSamplerDataModule
from heatsplats.trainers import BaseTrainer
from heatsplats.utils import combine_videos, show_video
from heatsplats.utils.typing import *


if __name__ == "__main__":

    experiment = "outputs/uv-texture-fitting/spot_triangulated@20260112-192635"

    extras_dict = {
        "renderer.point_batching": 512,
        "renderer.n_rotating_frames": 3,
        "renderer.integrator_config.type": "prb",
        "renderer.integrator_config.meta.max_depth": 2,
        "trainer.renderer_mega_kernel": False,
        "renderer.camera_config.tile_size_heatkernels": None,
        # "renderer.camera_config.img_width": 1024,
        # "renderer.camera_config.img_height": 1024,
        # "renderer.camera_config.tile_size_heatkernels": 16,
    }

    cfg_path = os.path.join(experiment, "configs/parsed.yaml")
    extras = [f"{k}={v}" for k, v in extras_dict.items()]
    cfg: ExperimentConfig = load_config(cfg_path, cli_args=extras)

    datamodule: MeshSamplerDataModule = heatsplats.find(cfg.data_type)(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")

    trainer: BaseTrainer = heatsplats.find(cfg.trainer_type)(
        cfg.trainer, datamodule, renderer_cfg=cfg.renderer
    )

    ckpt_name = os.path.join(cfg.trial_dir, "ckpts", cfg.optim.save_model_name)
    # trainer.model.load_torch(ckpt_name)
    trainer.model.load_numpy_npz(ckpt_name)

    rend_result = trainer.render_result(rotating_frames=8)
    rend_rings = trainer.render_kernel_rings(rotating_frames=8)
    combined_renderings = combine_videos(rend_result, rend_rings)

    print("Show rings rendering: show_video(rend_rings)")
    print("Show result rendering: show_video(rend_result)")
    print("Show combined rendering: show_video(combined_renderings)")
