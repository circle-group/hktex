import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import trimesh
import torch

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from heatsplats.modules import EigenAlboInterpolation
from heatsplats.utils import load_mesh, big_trimesh_pcl, compute_biharmonic_distance
import heatsplats.utils as utils
from heatsplats.modules import Mesh, HeatKernelTexture, EigenAlboInterpolation
from heatsplats.data.known_heat_vertex_colours import KnownHeatVertexColoursDataModule
from heatsplats.utils.typing import *

if __name__ == "__main__":
    model_cfg = {
        "weights": None,
        "n_sources": 400,
        "out_dim": 3,
        "kernel_dim": 32,
        "out_net": True,
        "normalize_colours": True,
    }
    eigalbo_config = {
        "k_eig": 256,
        "use_precomputed": True,
        "precompute_anisotropies": [1, 2.5, 5, 7.5, 10, 25, 50, 75, 100],
        "precompute_angles_every_deg": 30,
        "mesh_path": "../../objects/spot/spot_triangulated.obj",
        "precomputed_name": "eigen_albo",
        "distance_weighting": "gaussian_0.05",
    }

    tri_mesh = load_mesh(
        eigalbo_config["mesh_path"], merge_tex=True, bake_vert_colors=False
    )
    our_mesh = Mesh.from_trimesh(tri_mesh, device="cuda:0")
    model = HeatKernelTexture(model_cfg, our_mesh)
    eigalbo_interp = EigenAlboInterpolation(eigalbo_config, our_mesh)

    kernel_vert_idx = our_mesh.get_face_vertices(model.kernel_face_ids)
    kernel_barycentric_coords = our_mesh.cartesian_to_barycentric(
        model.kernel_locations, kernel_vert_idx
    )

    # display weights between a vertex and kernel centres
    selected_point_idx = 0
    subset_vertices_idxs = torch.randint(0, our_mesh.N_verts, (50,))
    pt_pos = our_mesh.verts[subset_vertices_idxs[selected_point_idx]]
    pt_tri = big_trimesh_pcl(pt_pos[None, ::], radius=0.03)

    w = eigalbo_interp.compute_biharmonic_weights(
        eigalbo_interp.ilbo_evec_vertices(subset_vertices_idxs),
        kernel_barycentric_coords,
        kernel_vert_idx,
    )[:, :-1]

    w_from_pt = w[:, selected_point_idx]
    cmap = plt.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=0, vmax=w_from_pt.max())
    colours = cmap(norm(w_from_pt.detach().cpu().numpy()))[:, :3]

    kernel_positions = model.kernel_locations
    kernels_tri = big_trimesh_pcl(kernel_positions, colours, radius=0.015)
    tri_mesh.visual = trimesh.visual.color.ColorVisuals(
        vertex_colors=np.zeros_like(tri_mesh.vertices) + np.array([125, 125, 125])
    )

    scene = trimesh.Scene([tri_mesh] + pt_tri + kernels_tri)
    print(
        f"w_min={w_from_pt.min()}, w_max={w_from_pt.max()}, w_mean={w_from_pt.mean()}"
    )

    # Visualise the effects of BH weighting on the heat diffusion.

    def diffuse(dm, normalise_colours, weights, device):
        idx_range = torch.arange(dm.n_sources, device=device)

        gt_colours = torch.zeros(
            [dm.n_sources, *dm.train_dataset.verts.shape], device=device
        )
        gt_colours[idx_range, dm.source_idxs, :] = dm.gt_splats["kernel_colours"].to(
            device
        )

        albo_weights = eigalbo_interp.interpolate_anisotropies(
            angles=torch.deg2rad(dm.gt_splats["angles"].to(device)),
            scales=dm.gt_splats["anisotropies"].to(device),
        )
        albo_evals, albo_evecs, mass = eigalbo_interp.albo_vertices(
            albo_weights=albo_weights
        )

        gt_colours = utils.heat_diffusion(
            gt_colours.to(device),
            mass,
            albo_evals,
            albo_evecs,
            dm.gt_splats["diff_times"].to(device),
            weights,
            at_vertices=True,
        )

        gt_colours = gt_colours.sum(dim=0)
        if normalise_colours:
            gt_colours = utils.normalise_colours(gt_colours)
        return gt_colours

    model_cfg["n_sources"] = 3

    model = HeatKernelTexture(model_cfg, our_mesh)

    datamodule = KnownHeatVertexColoursDataModule(
        cfg={
            "sample_all_vertices": True,
            "mesh_path": eigalbo_config["mesh_path"],
            "merge_tex": True,
        }
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.configure(
        n_sources=model.N_sources,
        diff_time_scaler_func=model._diff_time_scaler_func,
    )
    datamodule.bake_heat(eigalbo_interp, model.normalize_colours, device="cuda:0")

    # data = next(iter(datamodule.train_dataloader()))
    v_colours = datamodule.train_dataset.vcols

    tri_mesh_diffused = tri_mesh.copy()
    v_colours = (v_colours * 255).to(dtype=torch.uint8)
    v_colours = v_colours.squeeze().detach().cpu().numpy()
    tri_mesh_diffused.visual = trimesh.visual.ColorVisuals(
        tri_mesh_diffused, vertex_colors=v_colours
    )

    pts_kernel_dist = compute_biharmonic_distance(
        eigalbo_interp.ilbo_evec_vertices(),
        eigalbo_interp.ilbo_evec_vertices(datamodule.source_idxs),
        eigalbo_interp.iso_evals,
        pairwise=True,
    )
    weights = 1.0 / torch.clamp(pts_kernel_dist, min=1e-16)
    weights = weights / weights.sum(dim=0, keepdim=True)
    # sigma = 0.05  # Standard deviation of the Gaussian kernel
    # weights = torch.exp(-(pts_kernel_dist**2) / (2 * sigma**2))

    tri_mesh_weighted = tri_mesh.copy()
    v_colours_weighted = diffuse(datamodule, model.normalize_colours, weights, "cuda:0")
    v_colours_weighted = (v_colours_weighted * 255).to(dtype=torch.uint8)
    v_colours_weighted = v_colours_weighted.squeeze().detach().cpu().numpy()
    tri_mesh_weighted.visual = trimesh.visual.ColorVisuals(
        tri_mesh_weighted, vertex_colors=v_colours_weighted
    )

    print("here")
