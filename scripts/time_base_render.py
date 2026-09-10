import sys
import os
from pathlib import Path

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import time
import argparse
import numpy as np
import pandas as pd
import torch
import mitsuba as mi
import drjit as dr
from ray import tune

import hktex
from hktex.data import MeshSamplerDataModule
from hktex.trainers import BaseTrainer
from hktex.utils import load_config, seed_everything
from hktex.rendering.heat_kernels_renderer import HeatKernelsRenderer
from hktex.rendering.heat_kernels_renderer_knn import HeatKernelsRendererKNN

mi.set_variant("cuda_ad_rgb")


def safe_name(fname: str) -> str:
    return fname.replace("/", "_").replace("\\", "_").replace(".glb", "")


def find_ckpt_for_filename(benchmark_out_dir: str, fname: str) -> str | None:
    s = safe_name(fname)
    matches = list(Path(benchmark_out_dir).glob(f"**/output/{s}/ckpts/model.pt"))
    if not matches:
        return None
    return str(matches[0])


def _gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()


@torch.no_grad()
def timed_render_result_from_trainer(trainer, rotating_frames: int = 10):
    """
    Returns:
      out: mi.Bitmap | list[mi.Bitmap]
      timings: dict[str, float] in seconds
    """
    timings = {}

    _gpu_sync()
    t0 = time.perf_counter()

    # Part 1
    if trainer.cfg.use_knn_implementation:
        renderer = HeatKernelsRendererKNN(trainer.cfg.renderer)
    else:
        renderer = HeatKernelsRenderer(trainer.cfg.renderer)
    renderer.mega_kernel(
        trainer.cfg.renderer_mega_kernel, no_loops=True, no_opt_calls=True
    )

    _gpu_sync()
    t1 = time.perf_counter()
    timings["part1_renderer_init_s"] = t1 - t0

    # Part 2
    mi_mesh = renderer.mesh_to_mitsuba(
        trainer.datamodule.mesh, trainer.mesh, trainer.model, trainer.eigalbo_interp
    )
    if trainer.cfg.use_knn_implementation:
        trainer.model.prepare_kernels(trainer.mesh, trainer.eigalbo_interp, False)

    _gpu_sync()
    t2 = time.perf_counter()
    timings["part2_mesh_and_prepare_s"] = t2 - t1

    # Part 3
    if rotating_frames == 1:
        img = renderer.render(mi_mesh, denoise=True)
        out = mi.Bitmap(img).convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True,
        )
    else:
        out = renderer.rotating_video(mi_mesh, rotating_frames)

    _gpu_sync()
    t3 = time.perf_counter()
    timings["part3_render_s"] = t3 - t2
    timings["total_s"] = t3 - t0

    renderer.flush_cache()
    if trainer.cfg.use_knn_implementation:
        trainer.model.reset(trainer.eigalbo_interp)

    return out, timings


def load_datamodule_and_trainer_only(args, extras):
    # same config path flow as optimisation.main
    cfg = load_config(args.config, args.rendering_config, cli_args=extras, n_gpus=1)
    seed_everything(cfg.seed)

    datamodule: MeshSamplerDataModule = hktex.find(cfg.data_type)(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")

    cfg.trainer.density_controllers = []
    trainer: BaseTrainer = hktex.find(cfg.trainer_type)(
        cfg.trainer, datamodule, renderer_cfg=cfg.renderer
    )
    return cfg, datamodule, trainer


def infer_trainable(config):
    fname = config["filename"]
    mesh_root = config["root"]
    ckpt = config["ckpt_path"]

    args = argparse.Namespace(
        config=config["base_config_path"],
        rendering_config=config["rendering_config_path"],
        gpu="0",
        verbose=False,
    )

    mesh_path = os.path.join(mesh_root, fname)
    extras = [
        f"data.mesh_path={mesh_path}",
        f"trainer.eigen_albo.mesh_path={mesh_path}",
        f"trainer.eigen_albo.error_if_not_precomputed=True",
        "optim.iters=0",  # no training
        "optim.save_model=False",
        "optim.save_logs=False",
    ]

    cfg, datamodule, trainer = load_datamodule_and_trainer_only(args, extras)
    trainer.model.load_torch(ckpt)
    trainer.model.eval()

    # warmup (exclude from metrics)
    _ = timed_render_result_from_trainer(
        trainer, rotating_frames=config["rotating_frames"]
    )

    runs = []
    for _ in range(config["n_runs"]):
        _, t = timed_render_result_from_trainer(
            trainer, rotating_frames=config["rotating_frames"]
        )
        runs.append(t)

    # summarize
    keys = runs[0].keys()
    row = {"filename": fname, "ckpt_path": ckpt}
    for k in keys:
        vals = [r[k] for r in runs]
        row[f"{k}_mean"] = float(np.mean(vals))
        row[f"{k}_std"] = float(np.std(vals))
        row[f"{k}_min"] = float(np.min(vals))

    # Save individual CSV like run_benchmark_parallel.py
    individual_results_dir = os.path.join(config["output_dir"], "individual_results")
    os.makedirs(individual_results_dir, exist_ok=True)
    individual_csv = os.path.join(individual_results_dir, f"{safe_name(fname)}.csv")
    pd.DataFrame([row]).to_csv(individual_csv, index=False)

    tune.report(row)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/data2/objaverse")
    p.add_argument("--benchmark_out_dir", type=str, default="outputs/benchmark")
    p.add_argument(
        "--benchmark_csv", type=str, default="outputs/benchmark/benchmark_results.csv"
    )
    p.add_argument("--config", type=str, default="configs/multiview_hktex_knn_ray.yaml")
    p.add_argument("--rendering_config", type=str, default="configs/render.yaml")
    p.add_argument("--output_dir", type=str, default="outputs/time_base_render")
    p.add_argument("--rotating_frames", type=int, default=1)
    p.add_argument("--n_runs", type=int, default=1)
    p.add_argument("--max_concurrent_trials", type=int, default=1)
    p.add_argument("--gpus_per_trial", type=float, default=1.0)
    p.add_argument(
        "--max_items",
        type=int,
        default=100,
        help="Maximum number of non-error benchmark rows to process.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.benchmark_csv)
    # Keep only non-error rows
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]
    # Skip failed metric rows
    if "mse" in df.columns:
        df = df[np.isfinite(df["mse"])]
    print(f"Loaded {len(df)} entries, keeping {args.max_items}")
    df = df.head(args.max_items)

    items = []
    for _, r in df.iterrows():
        fname = r["filename"]
        ckpt = find_ckpt_for_filename(args.benchmark_out_dir, fname)
        if ckpt is None:
            continue
        items.append({"filename": fname, "ckpt_path": ckpt})
    print(f"Starting with {len(items)} items")
    search_space = {
        "item": tune.grid_search(items),
        "root": args.root,
        "base_config_path": os.path.abspath(args.config),
        "rendering_config_path": os.path.abspath(args.rendering_config),
        "output_dir": os.path.abspath(args.output_dir),
        "rotating_frames": args.rotating_frames,
        "n_runs": args.n_runs,
    }

    def wrapper(config):
        item = config.pop("item")
        config["filename"] = item["filename"]
        config["ckpt_path"] = item["ckpt_path"]
        return infer_trainable(config)

    tuner = tune.Tuner(
        tune.with_resources(wrapper, resources={"gpu": args.gpus_per_trial}),
        param_space=search_space,
        tune_config=tune.TuneConfig(max_concurrent_trials=args.max_concurrent_trials),
        run_config=tune.RunConfig(
            storage_path=os.path.abspath(args.output_dir), name="time_base_render"
        ),
    )

    results = tuner.fit()
    df_out = results.get_dataframe()
    os.makedirs(args.output_dir, exist_ok=True)

    output_csv = os.path.join(args.output_dir, "time_base_render_results.csv")
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            existing_df = existing_df.loc[
                :, ~existing_df.columns.str.contains("^Unnamed")
            ]
            df_out = pd.concat([existing_df, df_out], ignore_index=True)
            if "filename" in df_out.columns:
                df_out = df_out.drop_duplicates(subset=["filename"], keep="last")
        except Exception as e:
            print(f"Could not merge with existing results: {e}")

    df_out.to_csv(output_csv, index=False)
    print(f"Saved {len(df_out)} rows to {output_csv}")
