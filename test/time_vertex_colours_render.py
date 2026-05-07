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

import numpy as np
import pandas as pd
import torch
import mitsuba as mi
import drjit as dr
import trimesh
import yaml

from ray import tune

from heatsplats.utils import load_mesh
from heatsplats.rendering.vertex_colours_renderer import VertexColoursRenderer

mi.set_variant("cuda_ad_rgb")


def safe_name(fname: str) -> str:
    return fname.replace("/", "_").replace("\\", "_").replace(".glb", "")


def _gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()


def get_vertex_colours_size_bytes(mesh: trimesh.Trimesh) -> int:
    """Estimates the size of vertex colours in bytes using npz compression."""
    buffer = io.BytesIO()
    vc = None

    if getattr(mesh.visual, "kind", None) == "texture":
        try:
            temp_vis = mesh.visual.to_color()
            vc = np.array(temp_vis.vertex_colors[:, :3])
        except Exception:
            pass

    if vc is None:
        if (
            hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
            and len(mesh.visual.vertex_colors) > 0
        ):
            vc = np.array(mesh.visual.vertex_colors[:, :3])
        else:
            vc = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)

    np.savez_compressed(buffer, data=vc)
    return buffer.tell()


def prepare_vertex_colours_mesh(
    mesh_gt: trimesh.Trimesh, target_size_kb: float, match_size: bool
):
    mesh_vc = mesh_gt.copy()
    current_size_kb = get_vertex_colours_size_bytes(mesh_vc) / 1024.0

    if match_size and target_size_kb > 0 and current_size_kb < target_size_kb:
        original_material = mesh_vc.visual.material
        while current_size_kb < target_size_kb:
            estimated_next_vertices = 2 * len(mesh_vc.vertices) + len(mesh_vc.faces)
            if estimated_next_vertices > 1_500_000:
                break

            estimated_next_size_kb = current_size_kb * (
                estimated_next_vertices / max(1, len(mesh_vc.vertices))
            )
            if abs(estimated_next_size_kb - target_size_kb) > abs(
                current_size_kb - target_size_kb
            ):
                break

            if (
                not hasattr(mesh_vc.visual, "uv")
                or mesh_vc.visual.uv is None
                or len(mesh_vc.visual.uv) == 0
            ):
                break

            new_verts, new_faces, new_attrs = trimesh.remesh.subdivide(
                mesh_vc.vertices,
                mesh_vc.faces,
                vertex_attributes={"uv": mesh_vc.visual.uv},
            )

            new_mesh = trimesh.Trimesh(
                vertices=new_verts, faces=new_faces, process=False
            )
            new_mesh.visual = trimesh.visual.TextureVisuals(
                uv=new_attrs["uv"], material=original_material
            )
            mesh_vc = new_mesh
            current_size_kb = get_vertex_colours_size_bytes(mesh_vc) / 1024.0

    mesh_vc.visual = mesh_vc.visual.to_color()
    return mesh_vc, current_size_kb


@torch.no_grad()
def timed_vertex_colours_render(
    prepared_mesh: trimesh.Trimesh,
    renderer_config: dict,
    rotating_frames: int = 1,
):
    timings = {}

    _gpu_sync()
    t0 = time.perf_counter()
    renderer = VertexColoursRenderer(renderer_config)
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

    if hasattr(renderer, "flush_cache"):
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

    mesh_gt = load_mesh(mesh_path, merge_tex=False, bake_vert_colors=False)
    mesh_vc, final_size_kb = prepare_vertex_colours_mesh(
        mesh_gt, target_size_kb, config["match_size"]
    )

    # Warmup
    _ = timed_vertex_colours_render(
        mesh_vc,
        renderer_config,
        rotating_frames=rotating_frames,
    )

    runs = []
    for _ in range(config["n_runs"]):
        _, t = timed_vertex_colours_render(
            mesh_vc,
            renderer_config,
            rotating_frames=rotating_frames,
        )
        runs.append(t)

    row = {
        "filename": fname,
        "target_size_kb": target_size_kb,
        "storage_npz_kb_est": float(final_size_kb),
        "n_verts": len(mesh_vc.vertices),
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
        default="outputs/vertex_colours_benchmark/benchmark_results_vertex_colours.csv",
    )
    p.add_argument("--rendering_config", type=str, default="configs/rendering.yaml")
    p.add_argument(
        "--output_dir", type=str, default="outputs/time_vertex_colours_render"
    )
    p.add_argument("--rotating_frames", type=int, default=1)
    p.add_argument("--n_runs", type=int, default=1)
    p.add_argument("--max_items", type=int, default=100)
    p.add_argument("--max_concurrent_trials", type=int, default=1)
    p.add_argument("--gpus_per_trial", type=float, default=1.0)
    p.add_argument("--match_size", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.benchmark_csv)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]
    if "mse" in df.columns:
        df = df[np.isfinite(df["mse"])]

    # We fallback to use a default target 0.0 if not found, to support any CSV easily.
    target_key = "storage_npz_kb" if "storage_npz_kb" in df.columns else None

    df = df.head(args.max_items)
    items = []
    for _, r in df.iterrows():
        items.append(
            {
                "filename": str(r["filename"]),
                "target_size_kb": float(r[target_key]) if target_key else 0.0,
            }
        )
    print(f"Timing {len(items)} vertex colours entries")

    search_space = {
        "item": tune.grid_search(items),
        "root": args.root,
        "rendering_config_path": os.path.abspath(args.rendering_config),
        "output_dir": os.path.abspath(args.output_dir),
        "rotating_frames": args.rotating_frames,
        "n_runs": args.n_runs,
        "match_size": args.match_size,
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
            storage_path=os.path.abspath(args.output_dir),
            name="time_vertex_colours_render",
        ),
    )

    results = tuner.fit()
    df_out = results.get_dataframe()
    os.makedirs(args.output_dir, exist_ok=True)

    output_csv = os.path.join(args.output_dir, "time_vertex_colours_render_results.csv")
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
