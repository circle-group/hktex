"""
Script to run ImageGS fitting in parallel using Ray Tune.
Run this in the ImageGS environment.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
import trimesh
from PIL import Image
from ray import tune
import traceback
import shutil
import math
import io

# Add image-gs to path so we can import from it
current_dir = os.path.dirname(os.path.abspath(__file__))
image_gs_path = os.path.join(current_dir, "image-gs")


def get_texture_image(mesh):
    try:
        if mesh.visual.material.baseColorTexture is not None:
            return mesh.visual.material.baseColorTexture
    except AttributeError:
        if hasattr(mesh.visual.material, "image"):
            return mesh.visual.material.image
    return None


def get_uv_size_bytes(mesh):
    buffer = io.BytesIO()
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        np.savez_compressed(buffer, data=np.array(mesh.visual.uv))
    else:
        return 0
    return buffer.tell()


def fit_imagegs_trainable(config):
    # Imports inside trainable to avoid serialization issues and ensure env context
    import sys

    # Attempt to import gsplat before modifying sys.path to ensure we get the installed package
    # and not the local folder 'gsplat' inside 'image-gs' which might be picked up as a namespace package.
    try:
        import gsplat
    except ImportError:
        pass

    if "image_gs_path" in config and config["image_gs_path"] not in sys.path:
        sys.path.append(config["image_gs_path"])

    import torch
    from model import GaussianSplatting2D  # type: ignore

    row = config["row"]
    root = config["root"]
    output_dir = config["output_dir"]

    filename = row["filename"]
    target_kb = row["storage_npz_kb"]

    # Construct paths
    mesh_path = os.path.join(root, filename)

    # Create a safe name for the file
    parts = filename.split("/")
    if "glbs" in parts:
        idx = parts.index("glbs")
        safe_name = "_".join(parts[idx + 1 :]).replace(".glb", "")
    else:
        safe_name = os.path.basename(filename).replace(".", "_")

    trial_dir = tune.get_context().get_trial_dir()

    try:
        # 1. Load mesh and extract texture
        # We use trimesh here. Ensure trimesh is installed in ImageGS env.
        mesh = trimesh.load(mesh_path, process=False)

        # Handle scene vs mesh
        if hasattr(mesh, "graph"):
            # Simple concatenation for texture extraction if multiple geometries
            # This is a simplification; ideally we'd handle sub-meshes, but
            # for Objaverse single-object benchmark this usually works or we take the first.
            geometries = list(mesh.geometry.values())
            if len(geometries) > 0:
                mesh = geometries[0]

        # Calculate UV size cost
        uv_size_kb = get_uv_size_bytes(mesh) / 1024.0
        available_kb = target_kb - uv_size_kb

        texture = get_texture_image(mesh)
        if texture is None:
            raise ValueError("No texture found on mesh")

        # Save texture to a temporary file in the trial directory
        tex_filename = "texture.png"
        tex_path = os.path.join(trial_dir, tex_filename)

        if texture.mode not in ["RGB", "RGBA", "L"]:
            texture = texture.convert("RGBA")

        texture.save(tex_path)
        if not os.path.exists(tex_path):
            raise FileNotFoundError(
                f"Texture file was not saved correctly at {tex_path}"
            )

        # 2. Calculate Number of Gaussians
        # ImageGS storage per Gaussian (16-bit quantization):
        # pos(2) + scale(2) + rot(1) + feat(3) = 8 params
        # 8 params * 16 bits = 128 bits = 16 bytes

        # ImageGS storage per Gaussian:
        # pos(2) * 32 bits = 64 bits
        # scale(2) * 16 bits = 32 bits
        # rot(1) * 16 bits = 16 bits
        # feat(3) * 16 bits = 48 bits
        # Total = 160 bits = 20 bytes
        bytes_per_gaussian = 20

        if available_kb <= 0:
            print(
                f"Warning: UV size ({uv_size_kb:.2f} KB) exceeds target ({target_kb:.2f} KB). Using minimum Gaussians."
            )
            num_gaussians = 10
        else:
            target_bytes = available_kb * 1024
            # Multiply by 1.2 to account for compression and ensure we are slightly above the target size
            num_gaussians = int(math.ceil(target_bytes * 1.2 / bytes_per_gaussian))
            # Clamp to a reasonable minimum
            num_gaussians = max(10, num_gaussians)

        print(
            f"Processing {safe_name}: Target {target_kb:.2f} KB (UV: {uv_size_kb:.2f} KB) -> Available {available_kb:.2f} KB -> {num_gaussians} Gaussians"
        )

        # 3. Setup ImageGS Arguments
        # We mock the args object expected by GaussianSplatting2D
        class Args:
            def __init__(self):
                self.eval = False
                self.seed = 42
                self.device = "cuda"
                self.log_dir = os.path.join(trial_dir, "logs")
                self.log_level = "INFO"
                self.save_image_format = "png"
                self.save_plot_format = "png"
                self.vis_gaussians = False
                self.save_image_steps = 100000  # Don't save intermediate
                self.save_ckpt_steps = 100000  # Don't save intermediate
                self.eval_steps = 100000  # Don't eval

                # Data
                self.data_root = ""
                self.input_path = tex_path
                self.gamma = (
                    1.0  # Assuming texture is already in correct space or handled
                )
                self.downsample = False

                # Model / Optimization
                self.num_gaussians = num_gaussians
                self.disable_prog_optim = True  # Disable progressive for fixed budget
                self.max_steps = 5000  # Enough for convergence on 2D image?

                # Quantization (Matching user's 16-bit float storage)
                self.quantize = True
                self.pos_bits = 32
                self.scale_bits = 16
                self.rot_bits = 16
                self.feat_bits = 16

                # Other defaults from ImageGS
                self.topk = 1000  # Default
                self.disable_tiles = False
                self.init_scale = 1.0
                self.disable_topk_norm = False
                self.disable_inverse_scale = False
                self.disable_color_init = False
                self.l1_loss_ratio = 1.0
                self.l2_loss_ratio = 1.0
                self.ssim_loss_ratio = 0.0  # Can enable if needed
                self.pos_lr = 0.001
                self.scale_lr = 0.005
                self.rot_lr = 0.001
                self.feat_lr = 0.01
                self.disable_lr_schedule = False
                self.decay_ratio = 0.1
                self.check_decay_steps = 300
                self.max_decay_times = 3
                self.decay_threshold = 0.0002
                self.init_mode = "random"  # or 'gradient'
                self.init_random_ratio = 1.0

        args = Args()

        # 4. Run ImageGS
        model = GaussianSplatting2D(args)
        model.optimize()

        # Get actual NPZ size
        ckpt_dir = model.ckpt_dir
        npz_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".npz")]
        actual_npz_kb = 0.0
        if npz_files:
            latest_npz = max(
                npz_files, key=lambda x: int(x.split("-")[-1].split(".")[0])
            )
            npz_path = os.path.join(ckpt_dir, latest_npz)
            actual_npz_kb = os.path.getsize(npz_path) / 1024.0

        # 5. Render and Save Result
        # We render at the original resolution
        with torch.no_grad():
            # forward returns (image, time)
            # image is [C, H, W]
            rendered_tensor, _ = model.forward(
                texture.height, texture.width, model.tile_bounds
            )

            # Clamp and convert to PIL
            rendered_tensor = torch.clamp(rendered_tensor, 0.0, 1.0)
            # Assuming output is linear, apply gamma if needed, but usually texture is sRGB
            # ImageGS model usually outputs in the space of input images.

            ndarr = (rendered_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )
            res_image = Image.fromarray(ndarr)

            # Save to final output directory
            os.makedirs(output_dir, exist_ok=True)
            res_path = os.path.join(output_dir, f"{safe_name}.png")
            res_image.save(res_path)

            # Also save metadata
            meta_path = os.path.join(output_dir, f"{safe_name}_meta.csv")
            pd.DataFrame(
                [
                    {
                        "filename": filename,
                        "num_gaussians": num_gaussians,
                        "target_kb": target_kb,
                        "uv_size_kb": uv_size_kb,
                        "actual_npz_kb": actual_npz_kb,
                        "psnr": model.psnr_curr,
                    }
                ]
            ).to_csv(meta_path, index=False)

        tune.report({"done": True, "num_gaussians": num_gaussians})

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        traceback.print_exc()
        tune.report({"done": False, "error": str(e)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ImageGS benchmark fitting.")
    parser.add_argument(
        "--root", type=str, default="/data2/objaverse", help="Root of objaverse"
    )
    parser.add_argument(
        "--benchmark_csv", type=str, required=True, help="Path to benchmark results CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/imagegs_textures",
        help="Where to save fitted textures",
    )
    parser.add_argument("--max_concurrent_trials", type=int, default=4)
    parser.add_argument("--gpus_per_trial", type=float, default=0.5)

    args = parser.parse_args()

    if not os.path.exists(args.benchmark_csv):
        print(f"CSV not found: {args.benchmark_csv}")
        sys.exit(1)

    df = pd.read_csv(args.benchmark_csv)
    # Filter successful runs
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]

    output_dir = os.path.abspath(args.output_dir)
    # Create items for grid search
    items = []
    for idx, row in df.iterrows():
        filename = row["filename"]
        parts = filename.split("/")
        if "glbs" in parts:
            idx_part = parts.index("glbs")
            safe_name = "_".join(parts[idx_part + 1 :]).replace(".glb", "")
        else:
            safe_name = os.path.basename(filename).replace(".", "_")

        if os.path.exists(os.path.join(output_dir, f"{safe_name}.png")):
            continue
        items.append(row.to_dict())

    print(f"Found {len(items)} items to process.")

    tuner = tune.Tuner(
        tune.with_resources(
            fit_imagegs_trainable, resources={"gpu": args.gpus_per_trial}
        ),
        param_space={
            "row": tune.grid_search(items),
            "root": args.root,
            "output_dir": output_dir,
            "image_gs_path": image_gs_path,
        },
        tune_config=tune.TuneConfig(
            max_concurrent_trials=args.max_concurrent_trials,
        ),
        run_config=tune.RunConfig(
            storage_path=os.path.abspath("outputs/ray_imagegs"), name="imagegs_fitting"
        ),
    )

    results = tuner.fit()
    print("Fitting complete.")
