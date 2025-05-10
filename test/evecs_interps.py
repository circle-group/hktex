import sys
from pathlib import Path
import os

import trimesh.exchange

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from heatsplats.utils import (
    load_mesh,
    uniform_sampling,
    interpolate_barycentric_attr,
    big_trimesh_pcl,
)
from heatsplats.rendering.heat_kernels_renderer import HeatKernelsRenderer
from heatsplats.modules import Mesh, Model, EigenAlboInterpolation


if __name__ == "__main__":

    fname = "../objects/spot/spot_triangulated.obj"
    model_cfg = {
        "weights": None,
        "n_sources": 2,
        "out_dim": 3,
        "kernel_dim": 3,
        "out_net": False,
        "normalize_colours": False,
    }
    eigalbo_config = {
        "k_eig": 256,
        "use_precomputed": True,
        "precompute_anisotropies": [1, 2.5, 5, 7.5, 10, 25, 50, 75, 100],
        "precompute_angles_every_deg": 30,
        "mesh_path": "../objects/spot/spot_triangulated.obj",
        "precomputed_name": "eigen_albo",
    }
    device = "cuda:0"

    tri_mesh = load_mesh(fname, merge_tex=True, bake_vert_colors=True)
    our_mesh = Mesh.from_trimesh(tri_mesh, device=device)
    eigalbo_interp = EigenAlboInterpolation(eigalbo_config, our_mesh)

    angles = torch.deg2rad(torch.tensor([45.0, 10.0], device=device))
    anisotropies = torch.tensor([60, 5.2], device=device)
    kernel_face_ids = torch.tensor([2000, 4682], device=device)
    kernel_locations = torch.tensor(
        [[-0.3444, -0.5293, -0.0918], [-0.3068, 0.0106, 0.7313]], device=device
    )

    # Vertex colours ###################################################################

    albo_weights = eigalbo_interp.interpolate_anisotropies(
        angles=angles, scales=anisotropies
    )
    _, vert_evecs, _ = eigalbo_interp.albo_vertices(albo_weights=albo_weights)

    kernel_vert_idx = our_mesh.get_face_vertices(kernel_face_ids)
    barycentric_coords = our_mesh.cartesian_to_barycentric(
        kernel_locations, kernel_vert_idx
    )

    kernel_evecs, _ = eigalbo_interp.barycentric_albo_gaussians(
        albo_weights=albo_weights,
        barycentric_coords=barycentric_coords,
        vert_idx=kernel_vert_idx,
    )

    face_id, pts_barys = uniform_sampling(our_mesh.verts, our_mesh.faces, 500)
    pts_tri_vert_idx = our_mesh.get_face_vertices(face_id)

    _, pts_evecs, _ = eigalbo_interp.barycentric_albo_points(
        albo_weights=albo_weights,
        barycentric_coords=pts_barys,
        vert_idx=pts_tri_vert_idx,
    )

    kernel_id = 1
    evec_id = 200

    e_vert = vert_evecs[kernel_id, :, evec_id]
    e_pts = pts_evecs[kernel_id, :, evec_id]
    e_kernel = kernel_evecs[:, evec_id].unsqueeze(0)

    p_pts = interpolate_barycentric_attr(
        our_mesh.faces, face_id, pts_barys, our_mesh.verts
    )
    p_kernel = interpolate_barycentric_attr(
        our_mesh.faces, kernel_face_ids, barycentric_coords, our_mesh.verts
    )

    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=e_vert.min(), vmax=e_vert.max())
    c_vert = cmap(norm(e_vert.detach().cpu().numpy()))[:, :3]
    c_pts = cmap(norm(e_pts.detach().cpu().numpy()))[:, :3]
    c_kernel = cmap(norm(e_kernel.detach().cpu().numpy()))[:, :3]

    out_mesh = tri_mesh.copy()
    out_mesh.visual = trimesh.visual.ColorVisuals(out_mesh, vertex_colors=c_vert)

    pcl_pts = big_trimesh_pcl(p_pts, c_pts, radius=0.005)
    pcl_kernel = big_trimesh_pcl(p_kernel, c_pts, radius=0.015)
    pcl_pts_trim = trimesh.PointCloud(
        vertices=p_pts.cpu().detach().numpy(), colors=c_pts
    )

    scene = trimesh.Scene([out_mesh, *pcl_pts, *pcl_kernel])
    scene2 = trimesh.Scene([out_mesh, pcl_pts_trim, *pcl_kernel])

    # trimesh.exchange.export.export_scene(
    #     scene, "/homes/sf3018/Documents/geosplat/outputs/scene.obj"
    # )
