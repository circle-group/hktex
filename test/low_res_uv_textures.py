import sys
from pathlib import Path
import os
import numpy as np
import torch
import pandas as pd
import yaml
import torch.nn.functional as F

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import trimesh
import io
from PIL import Image
from typing import Optional, Union
import tempfile
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


def get_texture_image(
    mesh: trimesh.Trimesh,
) -> Union[Optional[Image.Image], Optional[str]]:
    """Gets the texture image from a mesh."""
    try:
        if mesh.visual.material.baseColorTexture is not None:
            return mesh.visual.material.baseColorTexture, "base"
    except AttributeError:
        if hasattr(mesh.visual.material, "image"):
            return mesh.visual.material.image, "image"
    return None, None


def get_image_size_bytes(image: Image.Image, format="png") -> int:
    """Gets the size of an image in bytes."""
    buffer = io.BytesIO()
    if format in ["png", "jpeg"]:
        image.save(buffer, format=format)
    elif format == "npz":
        np.savez_compressed(buffer, data=np.array(image))
    elif format == "pt":
        torch.save(torch.from_numpy(np.array(image)), buffer)
    else:
        raise ValueError(f"Unsupported format: {format}")
    return buffer.tell()


def get_uv_size_bytes(mesh: trimesh.Trimesh) -> int:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=np.array(mesh.visual.uv))
    return buffer.tell()


def downsample_image_to_target_size(
    image: Image.Image, target_size_kb: int, format="png"
):
    """Downsamples an image to a target size in KB."""
    target_size_bytes = target_size_kb * 1024

    min_scale = 0.0
    max_scale = 1.0
    best_image = None

    for _ in range(10):  # 10 iterations of binary search should be enough
        scale = (min_scale + max_scale) / 2.0
        if scale == 0:
            break

        # Ensure dimensions are at least 1x1 to avoid 0 dimension errors
        new_dims = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))

        resized_image = image.resize(new_dims, Image.LANCZOS)
        current_size_bytes = get_image_size_bytes(resized_image, format=format)

        if current_size_bytes > target_size_bytes:
            max_scale = scale
        else:
            min_scale = scale
            best_image = resized_image

    return best_image


def main(
    path: str, renderer_config: dict, output_dir: str, format="png", target_size_kb=222
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

    mesh = load_mesh(path, merge_tex=False)
    renderer = UVTextureRenderer(renderer_config)
    m_mi_gt = renderer.mesh_to_mitsuba(mesh)
    n_frames = renderer_config.get("n_rotating_frames", 3)
    gt_rend = renderer.rotating_video(m_mi_gt, n_frames=n_frames)

    texture, texture_type = get_texture_image(mesh)
    if texture is None:
        raise ValueError("Mesh has no texture.")
    print(f"Original shape: {texture.size}")
    texture.save(path[:-4] + f"_original.png", format="png")

    original_size_kb = get_image_size_bytes(texture, format=format) / 1024
    print(f"Original texture size: {original_size_kb:.2f} KB")

    uv_size_kb = get_uv_size_bytes(mesh) / 1024
    print(f"UV size: {uv_size_kb:.2f} KB")

    new_size_kb = original_size_kb
    available_kb = target_size_kb - uv_size_kb

    downsampled_image = None
    if available_kb > original_size_kb:
        downsampled_image = texture
    elif available_kb > 0:
        downsampled_image = downsample_image_to_target_size(
            texture, available_kb, format=format
        )

    if downsampled_image is None:
        if available_kb <= 0:
            print(
                f"Target size {target_size_kb:.2f} KB is smaller than UV size {uv_size_kb:.2f} KB. Using minimal texture."
            )
        else:
            print(
                f"Could not downsample image to {available_kb:.2f} KB. Using minimal texture."
            )
        downsampled_image = texture.resize((4, 4), Image.LANCZOS)

    if downsampled_image:
        # Save the downsampled image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=True) as temp_file:
            if format in ["png", "jpeg"]:
                downsampled_image.save(temp_file.name)
            elif format == "npz":
                np.savez_compressed(temp_file.name, data=np.array(downsampled_image))
            elif format == "pt":
                torch.save(
                    torch.from_numpy(np.array(downsampled_image)), temp_file.name
                )
            new_size_kb = os.path.getsize(temp_file.name) / 1024
            print(f"Downsampled texture size: {new_size_kb:.2f} KB")
            print(f"Downsampled texture and UV size: {new_size_kb + uv_size_kb:.2f} KB")
            # The file is not deleted automatically because delete=False
            # os.unlink(temp_file.name)

    if texture_type == "base":
        mesh.visual.material.baseColorTexture = downsampled_image
    elif texture_type == "image":
        mesh.visual.material.image = downsampled_image
    else:
        print("Unknown texture type; cannot assign downsampled image.")

    m_mi_res = renderer.mesh_to_mitsuba(mesh)
    result_rend = renderer.rotating_video(m_mi_res, n_frames=n_frames)

    gt = mibitmaps2torch(gt_rend)
    res = mibitmaps2torch(result_rend)

    mse = F.mse_loss(res, gt, reduction="mean").item()
    metrics = compute_all_image_metrics(res, gt)

    combined_rend = combine_videos(gt_rend, result_rend)
    renderings_dir = os.path.join(output_dir, "renderings_uv_low_res")
    os.makedirs(renderings_dir, exist_ok=True)
    save_video(combined_rend, os.path.join(renderings_dir, f"{safe_name}.mp4"))

    storage_npz_kb = new_size_kb + uv_size_kb

    row = {
        "filename": filename,
        "mse": mse,
        "n_kernels": 0,
        "storage_torch_kb": 0,
        "storage_npz_kb": storage_npz_kb,
        **metrics,
    }

    # Save individual CSV
    individual_results_dir = os.path.join(output_dir, "individual_results_uv_low_res")
    os.makedirs(individual_results_dir, exist_ok=True)
    individual_csv = os.path.join(individual_results_dir, f"{safe_name}.csv")
    pd.DataFrame([row]).to_csv(individual_csv, index=False)

    return result_rend, row


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Downsample a mesh texture to a target size."
    )
    parser.add_argument(
        "--mesh_path", type=str, default="none", help="Path to the mesh file."
    )
    parser.add_argument(
        "--target_size_kb", type=int, default=-1, help="Target size in KB."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="npz",
        choices=["png", "jpeg", "npz", "pt"],
        help="Image format.",
    )
    parser.add_argument(
        "--benchmark_csv", type=str, default=None, help="Path to benchmark results CSV."
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
    parser.add_argument("--output_dir", type=str, default="outputs/low_res_benchmark")
    args = parser.parse_args()

    with open(args.rendering_config, "r") as f:
        renderer_config = yaml.safe_load(f).get("renderer", {})

    default_filenames = [
        "/data2/objaverse/hf-objaverse-v1/glbs/000-087/0e708d1e0ce0447ba5637a5320f5729c.glb",
        "/data2/objaverse/hf-objaverse-v1/glbs/000-018/998d641ce1c74e44978a91fedc849905.glb",
        "/data2/objaverse/hf-objaverse-v1/glbs/000-074/5ecf9d1175ae405a9a073db305786411.glb",
        #
        # "/data2/objaverse/hf-objaverse-v1/glbs/000-101/818e088dc59f4a89bfea14cb46a4beca.glb",
        # "/data2/objaverse/hf-objaverse-v1/glbs/000-138/6713cc0cdad34f89a0256c5d2f68b7c1.glb",
        # "/data2/objaverse/hf-objaverse-v1/glbs/000-013/d79a32a512c64c5e93dc856864789a7e.glb",
        #
        "/data2/objaverse/hf-objaverse-v1/glbs/000-096/db5f9c28708142909b15212625a127f9.glb",
        "/data2/objaverse/hf-objaverse-v1/glbs/000-066/e7caba92073d4adba3477c21aa25e91f.glb",
        # "../objects/spot/spot_triangulated.obj",
        # "../objects/bob/bob_tri.obj",
        "../objects/human_tri/RUST_3d_Low1.obj",
        "../objects/cat_tri/12221_Cat_v1_l3.obj",
    ]

    filenames = []
    target_sizes = []

    if args.benchmark_csv:
        df = pd.read_csv(args.benchmark_csv)
        if "error" in df.columns:
            df = df[df["error"].isna()]

        for _, row in df.iterrows():
            filenames.append(os.path.join(args.root, row["filename"]))
            if args.target_size_kb != -1:
                target_sizes.append(args.target_size_kb)
            else:
                target_sizes.append(row["storage_npz_kb"])
    else:
        if args.mesh_path != "none":
            filenames = [args.mesh_path]
        else:
            filenames = default_filenames

        t_size = args.target_size_kb if args.target_size_kb != -1 else 3000
        target_sizes = [t_size] * len(filenames)

    output_csv = os.path.join(args.output_dir, "benchmark_results_uv_low_res.csv")
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if "filename" in existing_df.columns:
                processed_files = set(existing_df["filename"])
                new_filenames = []
                new_target_sizes = []
                for f, t in zip(filenames, target_sizes):
                    parts = f.split("/")
                    if "glbs" in parts:
                        idx = parts.index("glbs")
                        check_name = "/".join(parts[idx + 1 :])
                    else:
                        check_name = f

                    if check_name not in processed_files:
                        new_filenames.append(f)
                        new_target_sizes.append(t)

                print(
                    f"Skipping {len(filenames) - len(new_filenames)} already processed files."
                )
                filenames = new_filenames
                target_sizes = new_target_sizes
        except Exception as e:
            print(f"Could not filter existing results: {e}")

    renderings = []
    results = []
    for fname, t_size in zip(filenames, target_sizes):
        print(f"Processing {fname} with target size {t_size:.2f} KB...")
        try:
            rendering, row = main(
                fname,
                renderer_config,
                args.output_dir,
                format=args.format,
                target_size_kb=t_size,
            )
            renderings.extend(rendering)
            results.append(row)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            results.append({"filename": fname, "error": str(e)})

    if results:
        df = pd.DataFrame(results)
        if os.path.exists(output_csv):
            try:
                existing_df = pd.read_csv(output_csv)
                existing_df = existing_df.loc[
                    :, ~existing_df.columns.str.contains("^Unnamed")
                ]
                df = pd.concat([existing_df, df], ignore_index=True)
                if "filename" in df.columns:
                    df = df.drop_duplicates(subset=["filename"], keep="last")
            except Exception as e:
                print(f"Could not merge with existing results: {e}")
        df.to_csv(output_csv, index=False)
        print(f"Cumulative results saved to {output_csv}")

    print("show_video(renderings)")
