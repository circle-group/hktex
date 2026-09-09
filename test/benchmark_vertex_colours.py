import sys
from pathlib import Path
import os
import numpy as np
import torch
import pandas as pd
import yaml
import torch.nn.functional as F
import copy

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import trimesh
import io
import argparse
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

from hktex.utils import (
    load_mesh,
    show_video,
    save_video,
    combine_videos,
    mibitmaps2torch,
    compute_all_image_metrics,
)
from hktex.rendering.uv_texture_renderer import UVTextureRenderer
from hktex.rendering.vertex_colours_renderer import VertexColoursRenderer


def get_vertex_colours_size_bytes(mesh: trimesh.Trimesh) -> int:
    """Estimates the size of vertex colours in bytes using npz compression."""
    buffer = io.BytesIO()
    vc = None

    # Temporarily bake texture to colors. If we just use zeros for unbaked TextureVisuals,
    # np.savez_compressed crushes the size to almost nothing, causing massive overshoots.
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


def main(
    path: str,
    renderer_config: dict,
    output_dir: str,
    match_size: bool = False,
    target_size_kb: float = 0.0,
):
    # Derive safe_name and filename from path
    parts = path.split("/")
    if "glbs" in parts:
        idx = parts.index("glbs")
        rel_parts = parts[idx + 1 :]
        safe_name = "_".join(rel_parts).replace(".glb", "")
        filename = "/".join(rel_parts)
    else:
        safe_name = os.path.basename(path).replace(".", "_")
        filename = path

    # 1. Load GT mesh (with texture)
    mesh_gt = load_mesh(path, merge_tex=False, bake_vert_colors=False)

    # 2. Render GT
    uv_renderer = UVTextureRenderer(renderer_config)
    m_mi_gt = uv_renderer.mesh_to_mitsuba(mesh_gt)
    n_frames = renderer_config.get("n_rotating_frames", 3)
    gt_rend = uv_renderer.rotating_video(m_mi_gt, n_frames=n_frames)

    # 3. Load mesh for vertex colours
    # mesh_vc = load_mesh(path, merge_tex=True, bake_vert_colors=False)
    mesh_vc = mesh_gt.copy()

    if match_size and target_size_kb > 0:
        current_size_kb = get_vertex_colours_size_bytes(mesh_vc) / 1024.0
        print(
            f"Initial vertex colours size: {current_size_kb:.2f} KB, Target: {target_size_kb:.2f} KB"
        )

        if current_size_kb < target_size_kb:
            original_material = mesh_vc.visual.material
            while current_size_kb < target_size_kb:
                print(
                    f"Upsampling mesh... {current_size_kb:.2f}KB < {target_size_kb:.2f}KB"
                )

                # Predict next vertex count. Subdivision adds 1 vertex per edge.
                # By Euler's formula E ≈ V + F, so V_new ≈ 2*V + F
                estimated_next_vertices = 2 * len(mesh_vc.vertices) + len(mesh_vc.faces)
                if estimated_next_vertices > 1_500_000:
                    print(
                        f"Warning: Next subdivision would create ~{estimated_next_vertices} vertices. Stopping upsampling to prevent OOM crash."
                    )
                    break

                # Predict next size to avoid massive overshoots
                estimated_next_size_kb = current_size_kb * (
                    estimated_next_vertices / max(1, len(mesh_vc.vertices))
                )
                if abs(estimated_next_size_kb - target_size_kb) > abs(
                    current_size_kb - target_size_kb
                ):
                    print(
                        f"Stopping upsampling: next size (~{estimated_next_size_kb:.2f} KB) "
                        f"would be further from target ({target_size_kb:.2f} KB) than current ({current_size_kb:.2f} KB)."
                    )
                    break

                if (
                    not hasattr(mesh_vc.visual, "uv")
                    or mesh_vc.visual.uv is None
                    or len(mesh_vc.visual.uv) == 0
                ):
                    print(
                        "Warning: mesh has no UVs to preserve during subdivision. Stopping upsampling."
                    )
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
                print(f"New vertex colours size: {current_size_kb:.2f} KB")

    # 4. Bake vertex colours
    mesh_vc.visual = mesh_vc.visual.to_color()

    # 5. Calculate size of vertex colours
    vc_size_kb = get_vertex_colours_size_bytes(mesh_vc) / 1024.0
    print(f"Final vertex colours size: {vc_size_kb:.2f} KB")

    # 6. Render Vertex Colours
    vc_renderer = VertexColoursRenderer(renderer_config)
    m_mi_res = vc_renderer.mesh_to_mitsuba(mesh_vc)
    result_rend = vc_renderer.rotating_video(m_mi_res, n_frames=n_frames)

    gt = mibitmaps2torch(gt_rend)
    res = mibitmaps2torch(result_rend)

    mse = F.mse_loss(res, gt, reduction="mean").item()
    metrics = compute_all_image_metrics(res, gt)

    # Compute average number of foreground pixels per frame using a dedicated silhouette pass
    config_sil = copy.deepcopy(renderer_config)
    if "ground_plane_config" in config_sil:
        config_sil["ground_plane_config"]["activated"] = False
    if "integrator_config" in config_sil:
        config_sil["integrator_config"]["hide_emitters"] = True

    original_load_dict = mi.load_dict

    def patched_load_dict(d):
        def inject_rgba(node):
            if isinstance(node, dict):
                if node.get("type") == "hdrfilm":
                    node["pixel_format"] = "rgba"
                elif node.get("type") in ["perspective", "orthogonal", "thinlens"]:
                    if "film" in node:
                        inject_rgba(node["film"])
                    else:
                        node["film"] = {"type": "hdrfilm", "pixel_format": "rgba"}
                else:
                    for k, v in node.items():
                        inject_rgba(v)
            elif isinstance(node, list):
                for item in node:
                    inject_rgba(item)

        inject_rgba(d)
        return original_load_dict(d)

    mi.load_dict = patched_load_dict
    try:
        uv_renderer_sil = UVTextureRenderer(config_sil)
        m_mi_gt_sil = uv_renderer_sil.mesh_to_mitsuba(mesh_gt)

        azimuth = uv_renderer_sil.cfg.camera_config.azimuth_deg
        elevation = uv_renderer_sil.cfg.camera_config.elevation_deg
        total_fg_pixels = 0
        has_alpha = False

        for i in range(n_frames):
            uv_renderer_sil.change_camera_param(
                azimuth_deg=azimuth + (i / n_frames) * 360,
                elevation_deg=elevation,
            )
            frame_raw = uv_renderer_sil.render(m_mi_gt_sil, denoise=False)
            frame_torch = (
                frame_raw.torch() if hasattr(frame_raw, "torch") else frame_raw
            )
            if frame_torch.shape[-1] >= 4:
                has_alpha = True
                total_fg_pixels += (frame_torch[..., 3] > 0).float().sum().item()

        n_fg_pixels = total_fg_pixels / max(1, n_frames) if has_alpha else -1
    finally:
        mi.load_dict = original_load_dict

    combined_rend = combine_videos(gt_rend, result_rend)
    renderings_dir = os.path.join(output_dir, "renderings_vertex_colours")
    os.makedirs(renderings_dir, exist_ok=True)
    save_video(combined_rend, os.path.join(renderings_dir, f"{safe_name}.mp4"))

    n_verts = len(mesh_vc.vertices)
    row = {
        "filename": filename,
        "mse": mse,
        "n_kernels": 0,
        "n_verts": n_verts,
        "n_fg_pixels": n_fg_pixels,
        "storage_torch_kb": 0,
        "storage_npz_kb": vc_size_kb,
        **metrics,
    }

    # Save individual CSV
    individual_results_dir = os.path.join(
        output_dir, "individual_results_vertex_colours"
    )
    os.makedirs(individual_results_dir, exist_ok=True)
    individual_csv = os.path.join(individual_results_dir, f"{safe_name}.csv")
    pd.DataFrame([row]).to_csv(individual_csv, index=False)

    return result_rend, row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark vertex colours.")
    parser.add_argument(
        "--benchmark_csv",
        type=str,
        required=True,
        help="Path to benchmark results CSV.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/data2/objaverse",
        help="Root directory for objaverse",
    )
    parser.add_argument(
        "--rendering_config", type=str, default="configs/rendering.yaml"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/vertex_colours_benchmark"
    )
    parser.add_argument(
        "--match_size",
        action="store_true",
        help="Upsample mesh to match target size from CSV.",
    )
    args = parser.parse_args()

    with open(args.rendering_config, "r") as f:
        renderer_config = yaml.safe_load(f).get("renderer", {})

    df = pd.read_csv(args.benchmark_csv)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]

    output_csv = os.path.join(args.output_dir, "benchmark_results_vertex_colours.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for _, row in df.iterrows():
        filename = row["filename"]
        fname = os.path.join(args.root, filename)

        print(f"Processing {fname}...")
        try:
            target_size = 0.0
            if (
                args.match_size
                and "storage_npz_kb" in row
                and pd.notna(row["storage_npz_kb"])
            ):
                target_size = float(row["storage_npz_kb"])

            _, metrics_row = main(
                fname,
                renderer_config,
                args.output_dir,
                match_size=args.match_size,
                target_size_kb=target_size,
            )

            if "n_kernels" in row and pd.notna(row["n_kernels"]):
                metrics_row["n_kernels"] = int(row["n_kernels"])
                metrics_row["nk-nv"] = metrics_row["n_kernels"] - metrics_row["n_verts"]

            results.append(metrics_row)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            results.append({"filename": filename, "error": str(e)})

        # Update CSV progressively
        if results:
            df_out = pd.DataFrame(results)
            if os.path.exists(output_csv):
                try:
                    df_out.to_csv(output_csv, index=False)
                except Exception as e:
                    print(f"Could not merge with existing results: {e}")
            else:
                df_out.to_csv(output_csv, index=False)
