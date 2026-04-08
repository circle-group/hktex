import sys
from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np
import io
import torch
import mitsuba as mi

try:
    mi.set_variant("cuda_ad_rgb")
except Exception:
    pass

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import trimesh
from PIL import Image
from heatsplats.utils import load_mesh


def get_texture_image(mesh):
    try:
        if mesh.visual.material.baseColorTexture is not None:
            return mesh.visual.material.baseColorTexture
    except AttributeError:
        if hasattr(mesh.visual.material, "image"):
            return mesh.visual.material.image
    return None


def get_image_size_bytes(image: Image.Image, format="npz") -> int:
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


def get_uv_size_bytes(mesh) -> int:
    buffer = io.BytesIO()
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        np.savez_compressed(buffer, data=np.array(mesh.visual.uv))
        return buffer.tell()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Compute average original UV texture + UV coords storage cost."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV file containing filenames.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/data2/objaverse",
        help="Root directory for meshes.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="npz",
        choices=["png", "jpeg", "npz", "pt"],
        help="Format for texture size estimation.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at {args.csv}")
        return

    df = pd.read_csv(args.csv)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]

    print(f"Processing {len(df)} items from {args.csv}...")

    total_kb = 0.0
    count = 0

    for idx, row in df.iterrows():
        fname = row["filename"]
        mesh_path = os.path.join(args.root, fname)

        if not os.path.exists(mesh_path):
            continue

        try:
            mesh = load_mesh(mesh_path, merge_tex=False)

            uv_bytes = get_uv_size_bytes(mesh)

            texture = get_texture_image(mesh)
            tex_bytes = 0
            if texture is not None:
                tex_bytes = get_image_size_bytes(texture, format=args.format)

            size_kb = (uv_bytes + tex_bytes) / 1024.0
            total_kb += size_kb
            count += 1
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

    if count > 0:
        avg_kb = total_kb / count
        print(f"\nAverage storage cost (UV + Texture [{args.format}]): {avg_kb:.2f} KB")
        print(f"Total processed: {count}")
    else:
        print("\nNo meshes processed successfully.")


if __name__ == "__main__":
    main()
