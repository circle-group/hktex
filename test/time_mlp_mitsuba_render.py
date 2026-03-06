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

import heatsplats
from heatsplats.data import MeshSamplerDataModule
from heatsplats.trainers import BaseTrainer
from heatsplats.utils import load_config, seed_everything


mi.set_variant("cuda_ad_rgb")


def safe_name_mlp(fname: str) -> str:
    parts = fname.split("/")
    if "glbs" in parts:
        idx = parts.index("glbs")
        return "_".join(parts[idx + 1 :]).replace(".glb", "")
    return os.path.basename(fname).replace(".", "_")


def find_mlp_ckpt_for_filename(
    mlp_benchmark_out_dir: str, fname: str, encoding: str
) -> tuple[str | None, str | None]:
    sname = safe_name_mlp(fname)
    root = Path(mlp_benchmark_out_dir)

    patterns = [
        f"**/run_{encoding}/**/output/{sname}/ckpts/model.pt",
        f"**/output/{sname}/ckpts/model.pt",
    ]
    matches = []
    for p in patterns:
        matches.extend(root.glob(p))

    if not matches:
        return None, None

    ckpt_path = sorted(matches)[0]
    cfg_path = ckpt_path.parent.parent / "configs" / "parsed.yaml"
    if not cfg_path.exists():
        return str(ckpt_path), None
    return str(ckpt_path), str(cfg_path)


def _gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()


def resolve_mesh_path(mesh_root: str, fname: str) -> str:
    if os.path.isabs(fname) and os.path.isfile(fname):
        return fname

    root = mesh_root.rstrip("/\\")
    rel = fname.lstrip("/\\")
    prefix = "hf-objaverse-v1/glbs/"

    candidates = [os.path.join(root, rel)]

    if rel.startswith(prefix):
        candidates.append(os.path.join(root, rel[len(prefix) :]))
    else:
        candidates.append(os.path.join(root, prefix, rel))

    for p in candidates:
        if os.path.isfile(p):
            return p

    # Return the default join for a clear downstream error message.
    return candidates[0]


@torch.no_grad()
def timed_render_result_from_trainer(trainer, rotating_frames: int = 1):
    timings = {}

    _gpu_sync()
    t0 = time.perf_counter()

    # Part 1: renderer setup
    renderer = trainer._get_renderer()

    _gpu_sync()
    t1 = time.perf_counter()
    timings["part1_renderer_init_s"] = t1 - t0

    # Part 2: mesh_to_mitsuba (and texture binding)
    mi_mesh, _ = renderer.mesh_to_mitsuba(trainer.datamodule.mesh, trainer.model)

    _gpu_sync()
    t2 = time.perf_counter()
    timings["part2_mesh_and_prepare_s"] = t2 - t1

    # Part 3: render/convert
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
    return out, timings


def load_datamodule_and_trainer_only(args, extras):
    cfg = load_config(args.config, args.rendering_config, cli_args=extras, n_gpus=1)
    seed_everything(cfg.seed)

    datamodule: MeshSamplerDataModule = heatsplats.find(cfg.data_type)(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")

    trainer: BaseTrainer = heatsplats.find(cfg.trainer_type)(
        cfg.trainer, datamodule, renderer_cfg=cfg.renderer
    )
    return cfg, datamodule, trainer


def infer_trainable(config):
    fname = config["filename"]
    encoding = config["encoding"]
    mesh_root = config["root"]
    ckpt = config["ckpt_path"]

    args = argparse.Namespace(
        config=config["config_path"]
        if config.get("config_path") is not None
        else config["base_config_path"],
        rendering_config=config["rendering_config_path"],
        gpu="0",
        verbose=False,
    )

    mesh_path = resolve_mesh_path(mesh_root, fname)
    extras = [
        f"data.mesh_path={mesh_path}",
        "optim.iters=0",
        "optim.save_model=False",
        "optim.save_logs=False",
    ]
    # Keep laplacian precompute lookup tied to the current mesh path.
    if encoding == "laplacian":
        extras.append(f"trainer.network.encoding.mesh_path={mesh_path}")

    _, _, trainer = load_datamodule_and_trainer_only(args, extras)
    trainer.model.load_torch(ckpt)
    trainer.model.eval()

    # Warmup and initialize model-dependent scene state outside timed runs.
    _, _ = timed_render_result_from_trainer(
        trainer, rotating_frames=config["rotating_frames"]
    )
    _gpu_sync()

    runs = []
    for _ in range(config["n_runs"]):
        _, t = timed_render_result_from_trainer(
            trainer, rotating_frames=config["rotating_frames"]
        )
        runs.append(t)

    keys = runs[0].keys()
    row = {"filename": fname, "encoding": encoding, "ckpt_path": ckpt}
    for k in keys:
        vals = [r[k] for r in runs]
        row[f"{k}_mean"] = float(np.mean(vals))
        row[f"{k}_std"] = float(np.std(vals))
        row[f"{k}_min"] = float(np.min(vals))

    individual_results_dir = os.path.join(config["output_dir"], "individual_results")
    os.makedirs(individual_results_dir, exist_ok=True)
    individual_csv = os.path.join(
        individual_results_dir, f"{encoding}_{safe_name_mlp(fname)}.csv"
    )
    pd.DataFrame([row]).to_csv(individual_csv, index=False)

    tune.report(row)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/data2/objaverse")
    p.add_argument(
        "--mlp_benchmark_out_dir", type=str, default="outputs/mlp_benchmark_mitsuba"
    )
    p.add_argument(
        "--benchmark_csv",
        type=str,
        default="outputs/mlp_benchmark_mitsuba/benchmark_results_mlp_hash.csv",
    )
    p.add_argument(
        "--encoding",
        type=str,
        default=None,
        choices=["hash", "laplacian", "positional"],
        help="Optional encoding filter. Useful when benchmark_csv has multiple encodings "
        "or no encoding column.",
    )
    p.add_argument(
        "--config",
        type=str,
        default="/data/home/ck223/heatsplats/configs/uv_texture_mlp_fitting.yaml",
    )
    p.add_argument(
        "--base_timing_csv",
        type=str,
        default="outputs/timing_mitsuba_hs_ray_base/time_mitsuba_render_results.csv",
    )
    p.add_argument(
        "--rendering_config", type=str, default="configs/rendering.yaml"
    )
    p.add_argument("--output_dir", type=str, default="outputs/time_mlp_mitsuba_render")
    p.add_argument("--rotating_frames", type=int, default=1)
    p.add_argument("--n_runs", type=int, default=1)
    p.add_argument("--max_concurrent_trials", type=int, default=1)
    p.add_argument("--gpus_per_trial", type=float, default=1.0)
    p.add_argument(
        "--max_items",
        type=int,
        default=100,
        help="Maximum number of MLP rows to process after filtering.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.benchmark_csv)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]
    if "mse" in df.columns:
        df = df[np.isfinite(df["mse"])]

    if not os.path.exists(args.base_timing_csv):
        raise FileNotFoundError(f"Base timing CSV not found: {args.base_timing_csv}")
    df_base = pd.read_csv(args.base_timing_csv)
    if "filename" not in df_base.columns:
        raise ValueError("base_timing_csv must contain a 'filename' column")
    base_filenames = set(df_base["filename"].astype(str))

    if "filename" not in df.columns:
        raise ValueError("benchmark_csv must contain a 'filename' column")
    if "encoding" not in df.columns and args.encoding is None:
        raise ValueError(
            "benchmark_csv has no 'encoding' column; pass --encoding explicitly"
        )

    if args.encoding is not None:
        if "encoding" in df.columns:
            df = df[df["encoding"].astype(str) == args.encoding]

    df = df[df["filename"].astype(str).isin(base_filenames)]
    print(
        f"MLP benchmark rows after filtering with base timings: {len(df)}; "
        f"keeping first {args.max_items}"
    )
    df = df.head(args.max_items)

    items = []
    for _, r in df.iterrows():
        fname = str(r["filename"])
        encoding = str(r["encoding"]) if "encoding" in r else str(args.encoding)
        ckpt, cfg = find_mlp_ckpt_for_filename(
            args.mlp_benchmark_out_dir, fname, encoding
        )
        if ckpt is None:
            continue
        items.append(
            {
                "filename": fname,
                "encoding": encoding,
                "ckpt_path": ckpt,
                "config_path": cfg,
            }
        )
    print(f"Starting with {len(items)} items after checkpoint/config discovery")

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
        config["encoding"] = item["encoding"]
        config["ckpt_path"] = item["ckpt_path"]
        config["config_path"] = item["config_path"]
        return infer_trainable(config)

    tuner = tune.Tuner(
        tune.with_resources(wrapper, resources={"gpu": args.gpus_per_trial}),
        param_space=search_space,
        tune_config=tune.TuneConfig(max_concurrent_trials=args.max_concurrent_trials),
        run_config=tune.RunConfig(
            storage_path=os.path.abspath(args.output_dir), name="time_mlp_mitsuba_render"
        ),
    )

    results = tuner.fit()
    df_out = results.get_dataframe()
    os.makedirs(args.output_dir, exist_ok=True)

    output_csv = os.path.join(args.output_dir, "time_mlp_mitsuba_render_results.csv")
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            existing_df = existing_df.loc[
                :, ~existing_df.columns.str.contains("^Unnamed")
            ]
            df_out = pd.concat([existing_df, df_out], ignore_index=True)
            if {"filename", "encoding"}.issubset(df_out.columns):
                df_out = df_out.drop_duplicates(
                    subset=["filename", "encoding"], keep="last"
                )
            elif "filename" in df_out.columns:
                df_out = df_out.drop_duplicates(subset=["filename"], keep="last")
        except Exception as e:
            print(f"Could not merge with existing results: {e}")

    df_out.to_csv(output_csv, index=False)
    print(f"Saved {len(df_out)} rows to {output_csv}")
