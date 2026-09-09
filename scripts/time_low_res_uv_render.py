import sys
import os
from pathlib import Path

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import io
import time
import argparse
import tempfile

import numpy as np
import pandas as pd
import torch
import mitsuba as mi
import drjit as dr
import trimesh
import yaml

from PIL import Image
from ray import tune

from hktex.utils import load_mesh
from hktex.rendering.uv_texture_renderer import UVTextureRenderer

mi.set_variant("cuda_ad_rgb")


def safe_name(fname: str) -> str:
    return fname.replace("/", "_").replace("\\", "_").replace(".glb", "")


def _gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()


def get_texture_image(
    mesh: trimesh.Trimesh,
):
    try:
        if mesh.visual.material.baseColorTexture is not None:
            return mesh.visual.material.baseColorTexture, "base"
    except AttributeError:
        if hasattr(mesh.visual.material, "image"):
            return mesh.visual.material.image, "image"
    return None, None


def get_image_size_bytes(image: Image.Image, fmt: str = "png") -> int:
    buffer = io.BytesIO()
    if fmt in ["png", "jpeg"]:
        image.save(buffer, format=fmt)
    elif fmt == "npz":
        np.savez_compressed(buffer, data=np.array(image))
    elif fmt == "pt":
        torch.save(torch.from_numpy(np.array(image)), buffer)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return buffer.tell()


def get_uv_size_bytes(mesh: trimesh.Trimesh) -> int:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=np.array(mesh.visual.uv))
    return buffer.tell()


def downsample_image_to_target_size(
    image: Image.Image, target_size_kb: float, fmt="png"
):
    target_size_bytes = int(target_size_kb * 1024)
    min_scale = 0.0
    max_scale = 1.0
    best_image = None

    for _ in range(10):
        scale = (min_scale + max_scale) / 2.0
        if scale == 0:
            break
        new_dims = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resized_image = image.resize(new_dims, Image.LANCZOS)
        current_size_bytes = get_image_size_bytes(resized_image, fmt=fmt)
        if current_size_bytes > target_size_bytes:
            max_scale = scale
        else:
            min_scale = scale
            best_image = resized_image
    return best_image


def apply_low_res_texture(mesh: trimesh.Trimesh, target_size_kb: float, fmt: str):
    texture, texture_type = get_texture_image(mesh)
    if texture is None:
        raise ValueError("Mesh has no texture")

    uv_size_kb = get_uv_size_bytes(mesh) / 1024.0
    available_kb = target_size_kb - uv_size_kb
    original_size_kb = get_image_size_bytes(texture, fmt=fmt) / 1024.0

    if available_kb > original_size_kb:
        downsampled_image = texture
    elif available_kb > 0:
        downsampled_image = downsample_image_to_target_size(
            texture, available_kb, fmt=fmt
        )
    else:
        downsampled_image = None

    if downsampled_image is None:
        downsampled_image = texture.resize((4, 4), Image.LANCZOS)

    # Keep same byte-size estimate path used by low_res_uv_textures.py
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=True) as temp_file:
        if fmt in ["png", "jpeg"]:
            downsampled_image.save(temp_file.name)
        elif fmt == "npz":
            np.savez_compressed(temp_file.name, data=np.array(downsampled_image))
        elif fmt == "pt":
            torch.save(torch.from_numpy(np.array(downsampled_image)), temp_file.name)
        new_size_kb = os.path.getsize(temp_file.name) / 1024.0

    if texture_type == "base":
        mesh.visual.material.baseColorTexture = downsampled_image
    else:
        mesh.visual.material.image = downsampled_image

    return uv_size_kb, new_size_kb


@torch.no_grad()
def timed_low_res_uv_render(
    prepared_mesh: trimesh.Trimesh,
    renderer_config: dict,
    rotating_frames: int = 1,
):
    timings = {}

    _gpu_sync()
    t0 = time.perf_counter()
    renderer = UVTextureRenderer(renderer_config)
    renderer.mega_kernel(False)
    _gpu_sync()
    t1 = time.perf_counter()
    timings["part1_renderer_init_s"] = t1 - t0

    mi_mesh = renderer.mesh_to_mitsuba(prepared_mesh)
    _gpu_sync()
    t2 = time.perf_counter()
    timings["part2_mesh_and_prepare_s"] = t2 - t1

    if rotating_frames == 1:
        img = renderer.render(mi_mesh, denoise=True)
        out = mi.Bitmap(img).convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True,
        )
    else:
        out = renderer.rotating_video(mi_mesh, n_frames=rotating_frames)
    _gpu_sync()
    t3 = time.perf_counter()
    timings["part3_render_s"] = t3 - t2
    timings["total_s"] = t3 - t0

    renderer.flush_cache()
    return out, timings


def infer_trainable(config):
    fname = config["filename"]
    mesh_path = os.path.join(config["root"], fname)
    target_size_kb = float(config["target_size_kb"])

    with open(config["rendering_config_path"], "r") as f:
        renderer_config = yaml.safe_load(f).get("renderer", {})

    rotating_frames = int(config["rotating_frames"])
    if rotating_frames < 1:
        rotating_frames = int(renderer_config.get("n_rotating_frames", 1))

    # Prepare the low-res texture once; treated as input data (not timed).
    mesh = load_mesh(mesh_path, merge_tex=False)
    uv_size_kb, tex_size_kb = apply_low_res_texture(
        mesh, target_size_kb, fmt=config["format"]
    )

    # Warmup
    _ = timed_low_res_uv_render(
        mesh,
        renderer_config,
        rotating_frames=rotating_frames,
    )

    runs = []
    for _ in range(config["n_runs"]):
        _, t = timed_low_res_uv_render(
            mesh,
            renderer_config,
            rotating_frames=rotating_frames,
        )
        runs.append(t)

    row = {
        "filename": fname,
        "target_size_kb": target_size_kb,
        "uv_size_kb": float(uv_size_kb),
        "texture_size_kb": float(tex_size_kb),
        "storage_npz_kb_est": float(uv_size_kb + tex_size_kb),
    }
    for k in runs[0].keys():
        vals = [r[k] for r in runs]
        row[f"{k}_mean"] = float(np.mean(vals))
        row[f"{k}_std"] = float(np.std(vals))
        row[f"{k}_min"] = float(np.min(vals))

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
    p.add_argument(
        "--benchmark_csv",
        type=str,
        default="/data/hktex/benchmark_low_mem/benchmark_results_uv_low_res.csv",
    )
    p.add_argument(
        "--rendering_config",
        type=str,
        default="/data/hktex/configs/rendering.yaml",
    )
    p.add_argument("--output_dir", type=str, default="outputs/time_low_res_uv_render")
    p.add_argument(
        "--format",
        type=str,
        default="npz",
        choices=["png", "jpeg", "npz", "pt"],
        help="Format used for texture-byte estimation/downsampling target matching.",
    )
    p.add_argument(
        "--rotating_frames",
        type=int,
        default=1,
        help="If <1, uses renderer.n_rotating_frames from rendering config.",
    )
    p.add_argument("--n_runs", type=int, default=1)
    p.add_argument("--max_items", type=int, default=100)
    p.add_argument("--max_concurrent_trials", type=int, default=1)
    p.add_argument("--gpus_per_trial", type=float, default=1.0)
    args = p.parse_args()

    df = pd.read_csv(args.benchmark_csv)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]
    if "mse" in df.columns:
        df = df[np.isfinite(df["mse"])]
    if "filename" not in df.columns or "storage_npz_kb" not in df.columns:
        raise ValueError("benchmark_csv must contain 'filename' and 'storage_npz_kb'")

    df = df.head(args.max_items)
    items = [
        {"filename": str(r["filename"]), "target_size_kb": float(r["storage_npz_kb"])}
        for _, r in df.iterrows()
    ]
    print(f"Timing {len(items)} low-res UV entries")

    search_space = {
        "item": tune.grid_search(items),
        "root": args.root,
        "rendering_config_path": os.path.abspath(args.rendering_config),
        "output_dir": os.path.abspath(args.output_dir),
        "format": args.format,
        "rotating_frames": args.rotating_frames,
        "n_runs": args.n_runs,
    }

    def wrapper(config):
        item = config.pop("item")
        config["filename"] = item["filename"]
        config["target_size_kb"] = item["target_size_kb"]
        return infer_trainable(config)

    tuner = tune.Tuner(
        tune.with_resources(wrapper, resources={"gpu": args.gpus_per_trial}),
        param_space=search_space,
        tune_config=tune.TuneConfig(max_concurrent_trials=args.max_concurrent_trials),
        run_config=tune.RunConfig(
            storage_path=os.path.abspath(args.output_dir), name="time_low_res_uv_render"
        ),
    )

    results = tuner.fit()
    df_out = results.get_dataframe()
    os.makedirs(args.output_dir, exist_ok=True)

    output_csv = os.path.join(args.output_dir, "time_low_res_uv_render_results.csv")
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
