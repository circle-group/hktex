"""
Script to evaluate ImageGS fitted textures.
Run this in the HKTex environment.
"""

import sys
import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import mitsuba as mi
from pathlib import Path
from PIL import Image
import numpy as np
import yaml

try:
    script_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(script_dir))
except NameError:
    pass

mi.set_variant("cuda_ad_rgb")

from hktex.utils import (
    load_mesh,
    mibitmaps2torch,
    compute_all_image_metrics,
    save_video,
    combine_videos,
)
from hktex.rendering.uv_texture_renderer import UVTextureRenderer


def main():
    parser = argparse.ArgumentParser(description="Evaluate ImageGS textures.")
    parser.add_argument("--root", type=str, default="/data2/objaverse")
    parser.add_argument(
        "--benchmark_csv",
        type=str,
        required=True,
        help="Path to original benchmark CSV",
    )
    parser.add_argument(
        "--textures_dir",
        type=str,
        required=True,
        help="Directory containing ImageGS textures",
    )
    parser.add_argument(
        "--output_csv", type=str, default="outputs/imagegs_benchmark_results.csv"
    )
    parser.add_argument(
        "--save_renderings", action="store_true", help="Save comparison videos"
    )
    parser.add_argument(
        "--renderings_dir", type=str, default="outputs/imagegs_renderings"
    )
    parser.add_argument(
        "--rendering_config", type=str, default="configs/render.yaml"
    )

    args = parser.parse_args()

    # Load original benchmark data to get filenames
    df = pd.read_csv(args.benchmark_csv)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]

    # Setup renderer
    with open(args.rendering_config, "r") as f:
        config = yaml.safe_load(f)

    renderer_config = config.get("renderer", config)
    n_frames = renderer_config.get("n_rotating_frames", 5)

    renderer = UVTextureRenderer(renderer_config)

    results = []

    for idx, row in df.iterrows():
        filename = row["filename"]
        mesh_path = os.path.join(args.root, filename)

        # Construct safe name to find the texture
        parts = filename.split("/")
        if "glbs" in parts:
            idx = parts.index("glbs")
            safe_name = "_".join(parts[idx + 1 :]).replace(".glb", "")
        else:
            safe_name = os.path.basename(filename).replace(".", "_")

        tex_path = os.path.join(args.textures_dir, f"{safe_name}.png")

        if not os.path.exists(tex_path):
            print(f"Texture not found for {filename} at {tex_path}, skipping.")
            continue

        print(f"Evaluating {filename}...")

        try:
            # 1. Load Mesh
            mesh = load_mesh(mesh_path, merge_tex=False)

            # 2. Render GT (Original Texture)
            mi_mesh_gt = renderer.mesh_to_mitsuba(mesh)
            gt_rend = renderer.rotating_video(mi_mesh_gt, n_frames=n_frames)
            gt_tensor = mibitmaps2torch(gt_rend)

            # 3. Load ImageGS Texture and Assign
            imagegs_tex = Image.open(tex_path)

            # Assign to mesh
            # We need to handle how trimesh stores texture
            if hasattr(mesh.visual.material, "baseColorTexture"):
                mesh.visual.material.baseColorTexture = imagegs_tex
            elif hasattr(mesh.visual.material, "image"):
                mesh.visual.material.image = imagegs_tex
            else:
                # Fallback if material is weird, force create simple material
                # This might happen if original mesh had vertex colors but we want to force texture
                # But here we assume original had texture since we fit to it.
                pass

            # 4. Render Result
            mi_mesh_res = renderer.mesh_to_mitsuba(mesh)
            res_rend = renderer.rotating_video(mi_mesh_res, n_frames=n_frames)
            res_tensor = mibitmaps2torch(res_rend)

            # 5. Compute Metrics
            mse = F.mse_loss(res_tensor, gt_tensor, reduction="mean").item()
            metrics = compute_all_image_metrics(res_tensor, gt_tensor)

            # Get metadata if available (for num_gaussians)
            meta_path = os.path.join(args.textures_dir, f"{safe_name}_meta.csv")
            num_gaussians = 0
            actual_npz_kb = 0.0
            uv_size_kb = 0.0
            if os.path.exists(meta_path):
                try:
                    meta_df = pd.read_csv(meta_path)
                    num_gaussians = meta_df.iloc[0]["num_gaussians"]
                    if "actual_npz_kb" in meta_df.columns:
                        actual_npz_kb = meta_df.iloc[0]["actual_npz_kb"]
                    if "uv_size_kb" in meta_df.columns:
                        uv_size_kb = meta_df.iloc[0]["uv_size_kb"]
                except:
                    pass

            res_row = {
                "filename": filename,
                "mse": mse,
                "n_kernels": num_gaussians,  # Mapping gaussians to kernels column for comparison
                "target_kb": row["storage_npz_kb"],  # Target storage
                "storage_npz_kb": actual_npz_kb + uv_size_kb,
                **metrics,
            }
            results.append(res_row)

            if args.save_renderings:
                os.makedirs(args.renderings_dir, exist_ok=True)
                combined = combine_videos(gt_rend, res_rend)
                save_video(
                    combined, os.path.join(args.renderings_dir, f"{safe_name}.mp4")
                )

        except Exception as e:
            print(f"Error evaluating {filename}: {e}")

    # Save results
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(args.output_csv, index=False)
        print(f"Saved evaluation results to {args.output_csv}")

        # Print summary
        print("\nSummary:")
        print(res_df[["psnr", "ssim", "lpips"]].mean())


if __name__ == "__main__":
    main()
