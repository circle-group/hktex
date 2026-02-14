import sys
from pathlib import Path
import os
import numpy as np
import torch

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

from heatsplats.utils import load_mesh, show_video
from heatsplats.rendering.uv_texture_renderer import UVTextureRenderer


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

        new_dims = (int(image.width * scale), int(image.height * scale))
        if new_dims[0] == 0 or new_dims[1] == 0:
            break

        resized_image = image.resize(new_dims, Image.LANCZOS)
        current_size_bytes = get_image_size_bytes(resized_image, format=format)

        if current_size_bytes > target_size_bytes:
            max_scale = scale
        else:
            min_scale = scale
            best_image = resized_image

    print(new_dims)
    return best_image


def main(path: str, format="png", target_size_kb=222):
    mesh = load_mesh(path, merge_tex=False)

    texture, texture_type = get_texture_image(mesh)
    if texture is None:
        print("Mesh has no texture.")
        return
    print(f"Original shape: {texture.size}")
    texture.save(path[:-4] + f"_original.png", format="png")

    original_size_kb = get_image_size_bytes(texture, format=format) / 1024
    print(f"Original texture size: {original_size_kb:.2f} KB")

    uv_size_kb = get_uv_size_bytes(mesh) / 1024
    print(f"UV size: {uv_size_kb:.2f} KB")

    downsampled_image = downsample_image_to_target_size(
        texture, target_size_kb - uv_size_kb, format=format
    )

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
    else:
        print("Could not downsample image to the target size.")

    if texture_type == "base":
        mesh.visual.material.baseColorTexture = downsampled_image
    elif texture_type == "image":
        mesh.visual.material.image = downsampled_image
    else:
        print("Unknown texture type; cannot assign downsampled image.")

    renderer = UVTextureRenderer({})
    m_mi = renderer.mesh_to_mitsuba(mesh)
    return renderer.rotating_video(m_mi, n_frames=3)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Downsample a mesh texture to a target size."
    )
    parser.add_argument(
        "--mesh_path", type=str, default="none", help="Path to the mesh file."
    )
    parser.add_argument(
        "--target_size_kb", type=int, default=3000, help="Target size in KB."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="npz",
        choices=["png", "jpeg", "npz", "pt"],
        help="Image format.",
    )
    args = parser.parse_args([])

    filenames = [
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

    if args.mesh_path != "none":
        filenames = [args.mesh_path]

    renderings = []
    for fname in filenames:
        print(f"Processing {fname}...")
        rendering = main(fname, format=args.format, target_size_kb=args.target_size_kb)
        renderings.extend(rendering)

    print("show_video(renderings)")
