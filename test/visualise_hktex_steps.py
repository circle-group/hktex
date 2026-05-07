import sys
from pathlib import Path
import os
import types

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import torch
import trimesh
import numpy as np
import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant("cuda_ad_rgb")

from heatsplats.utils import (
    load_mesh,
    combine_images,
    show_image,
    rescaled_soft_step,
)
from heatsplats.rendering.heat_kernels_renderer_knn import HeatKernelsRendererKNN
from heatsplats.modules import Mesh, HeatKernelTextureKNN, EigenAlboInterpolationKNN


def apply_colormap(img_tensor, cmap_name="inferno"):
    """Applies a colormap to a grayscale Mitsuba tensor."""
    # Convert to numpy
    if isinstance(img_tensor, mi.TensorXf):
        img_np = img_tensor.torch().cpu().numpy()
    else:
        img_np = img_tensor.detach().cpu().numpy()

    # Take mean to get scalar intensity (in case it's RGB grayscale)
    if img_np.shape[-1] == 3:
        scalar = np.mean(img_np, axis=2)
    else:
        scalar = img_np.squeeze(-1)

    # Normalize roughly to [0,1] for visualization if needed,
    # but heat kernels are generally [0,1] already.
    # scalar = np.clip(scalar, 0, 1)

    cmap = plt.get_cmap(cmap_name)
    colored = cmap(scalar)[:, :, :3]  # RGBA -> RGB
    return mi.TensorXf(colored)


if __name__ == "__main__":
    fname = "../objects/spot/spot_triangulated.obj"
    device = "cuda:0"

    print(f"Loading mesh from {fname}...")
    tri_mesh = load_mesh(
        fname, merge_tex=True, bake_vert_colors=True, normalise_size=True
    )
    our_mesh = Mesh.from_trimesh(tri_mesh, device=device)

    k_aniso = 128
    k_iso = 64
    tau = 0.5
    dt = 0.05

    # Configuration
    model_cfg = {
        "n_sources": 3,
        "out_dim": 3,
        "kernel_dim": 3,
        "out_net": False,
        "normalize_colours": False,
        "knn_outer_k": 32,
        "knn_inner_k": 32,  # Accumulate all for visualization
        "init_kernel_edge_type": "uniform",
        "diff_time": dt,
    }

    eigalbo_config = {
        "k_eig": k_aniso,
        "use_precomputed": False,
        "precompute_anisotropies": [5, 15, 30, 60, 100],
        "precompute_angles_every_deg": 30,
        "mesh_path": fname,
        "precomputed_name": "eigen_albo",
        "distance_weighting": "gaussian_0.1",
        "normalise_evals": False,
        "knn_embedding_dim": k_iso,
        "use_weighting": True,
    }

    model = HeatKernelTextureKNN(model_cfg, our_mesh)
    eigalbo_interp = EigenAlboInterpolationKNN(eigalbo_config, our_mesh)

    # Disable activations for direct control
    model._angle_act = lambda x: torch.deg2rad(x)
    model._anis_act = lambda x: x
    model._sharpness_act = lambda x: x
    model._thresholds_act = lambda x: x
    model._colour_act = lambda x: x

    # Manually set parameters for 2 kernels
    with torch.no_grad():
        model._mean_colour.data.fill_(0.0)
        model._angles[:] = 0.0
        model._anisotropies[:] = 30.0
        model._sharpnesses[:] = 100.0
        model._thresholds[:] = tau
        model._kernel_colours[:] = 0.0

        # Kernel 0 (Red)
        model._kernel_colours[0] = torch.tensor([1.0, 0.0, 0.0], device=device)

        # Kernel 1 (Green) nearby
        model._kernel_colours[1] = torch.tensor([0.0, 1.0, 0.0], device=device)

        # Position kernels
        idx0 = 2000
        idx1 = 4682

        face_verts = our_mesh.get_face_vertices(
            torch.tensor([idx0, idx1], device=device)
        )
        barys = torch.tensor([[0.5, 0.5, 0], [0.5, 0.5, 0]], device=device)
        locs = our_mesh.barycentric_to_cartesian(barys, face_verts)

        model._kernel_locations[0] = locs[0]
        model._kernel_locations[1] = locs[1]
        model._kernel_face_ids[0] = idx0
        model._kernel_face_ids[1] = idx1

    renderer = HeatKernelsRendererKNN(
        {
            "camera_config": {
                "azimuth_deg": -90,
                "camera_distance": 3.5,
                "img_width": 512,
                "img_height": 512,
            },
            "point_batching": 1024 * 16,
        }
    )
    renderer.mega_kernel(False)

    # Initialize everything
    model.prepare_kernels(our_mesh, eigalbo_interp)
    mi_mesh = renderer.mesh_to_mitsuba(tri_mesh, our_mesh, model, eigalbo_interp)

    # Save original method
    original_method = model.diffuse_heat_kernels

    # --- Step 0: Un-normalised Heat Kernel ---
    def patch_step0(self, eigalbo_interp, pts_info):
        heat_qk, heat_qk_norm = eigalbo_interp.diffuse_heat(
            pts_info["albo_evals"],
            pts_info["albo_evecs"],
            pts_info["indices"],
            weights=None,
        )
        # Use un-normalized heat kernel
        val = heat_qk
        val = val.sum(dim=1, keepdim=True)
        return val.repeat(1, self.out_dim), None, None, None

    print("Rendering Step 0...")
    model.diffuse_heat_kernels = types.MethodType(patch_step0, model)
    img_0 = renderer.render(mi_mesh, denoise=True)
    # Normalize for visualization as raw values can be > 1
    t_0 = img_0.torch()
    print(
        f"Step 0 (Pre-normalization) Range: [{t_0.min().item():.6e}, {t_0.max().item():.6e}]"
    )
    img_0_norm = t_0 / (t_0.max() + 1e-8)
    img_0 = apply_colormap(mi.TensorXf(img_0_norm), "viridis")

    # --- Step 1: Normalised Heat Kernel ---
    def patch_step1(self, eigalbo_interp, pts_info):
        heat_qk, heat_qk_norm = eigalbo_interp.diffuse_heat(
            pts_info["albo_evals"],
            pts_info["albo_evecs"],
            pts_info["indices"],
            weights=None,
        )
        val = heat_qk_norm if heat_qk_norm is not None else heat_qk
        val = val.sum(dim=1, keepdim=True)
        return val.repeat(1, self.out_dim), None, None, None

    print("Rendering Step 1...")
    model.diffuse_heat_kernels = types.MethodType(patch_step1, model)
    img_1 = renderer.render(mi_mesh, denoise=True)
    img_1 = apply_colormap(img_1, "plasma")

    # --- Step 2: Biharmonic Distance Weighting ---
    def patch_step2(self, eigalbo_interp, pts_info):
        heat_qk, heat_qk_norm = eigalbo_interp.diffuse_heat(
            pts_info["albo_evals"],
            pts_info["albo_evecs"],
            pts_info["indices"],
            weights=pts_info["weights"],
        )
        val = heat_qk_norm if heat_qk_norm is not None else heat_qk
        val = val.sum(dim=1, keepdim=True)
        return val.repeat(1, self.out_dim), None, None, None

    print("Rendering Step 2...")
    model.diffuse_heat_kernels = types.MethodType(patch_step2, model)
    img_2 = renderer.render(mi_mesh, denoise=True)
    img_2 = apply_colormap(img_2, "plasma")

    # --- Step 3: Kernel Filtering ---
    def patch_step3(self, eigalbo_interp, pts_info):
        # Reuse logic from original method but skip color
        # We need the filtered values
        # This basically calls the original but returns filtered sum instead of colors
        # For simplicity, let's just use the original method logic partially
        heat_qk, heat_qk_norm = eigalbo_interp.diffuse_heat(
            pts_info["albo_evals"],
            pts_info["albo_evecs"],
            pts_info["indices"],
            weights=pts_info["weights"],
        )
        diffused = heat_qk_norm if heat_qk_norm is not None else heat_qk
        filtered = self.kernel_filter_func(
            diffused,
            epsilon=self.thresholds[pts_info["indices"]],
            sharpness=self.sharpnesses[pts_info["indices"]],
        )
        val = filtered.sum(dim=1, keepdim=True)
        return val.repeat(1, self.out_dim), None, None, None

    print("Rendering Step 3...")
    model.diffuse_heat_kernels = types.MethodType(patch_step3, model)
    img_3 = renderer.render(mi_mesh, denoise=True)
    img_3 = apply_colormap(img_3, "plasma")

    # --- Step 4: Colour Formation ---
    print("Rendering Step 4...")
    model.diffuse_heat_kernels = original_method  # Restore
    img_4 = renderer.render(mi_mesh, denoise=True)
    # img_4 is already colored properly

    combined = combine_images(img_0, img_1, img_2, img_3, img_4)
    combined.write(
        f"outputs/diff_stages/k_aniso={k_aniso}-k_iso={k_iso}-tau={tau}-dt={dt}.png"
    )
    print("Done.")
    print(
        "Columns: Un-normalised HK -> Normalised HK -> Weighted HK -> Filtered -> Final Colour"
    )
    print("Show with: show_image(combined)")
