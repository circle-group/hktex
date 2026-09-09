import omegaconf

import numpy as np
import torch
import trimesh

from hktex.modules import EigenAlboInterpolation
from hktex.utils import (
    load_mesh,
    heat_diffusion,
)
from hktex.utils.typing import *

if __name__ == "__main__":
    mesh_path = "../objects/spot/spot_triangulated.ply"
    mesh = load_mesh(mesh_path, show=False)

    # v, f = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    # v, f = trimesh.remesh.subdivide(v, f)
    # mesh = trimesh.Trimesh(v, f)

    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    fnorm = np.array(mesh.face_normals)

    hk_config = omegaconf.OmegaConf.create({"k_eig": 256, "mesh_path": mesh_path})
    pca_eigen_albo = EigenAlboInterpolation(hk_config, verts, faces, fnorm)
    device = pca_eigen_albo.device

    # albo_evals, albo_evecs, mass = pca_eigen_albo.get_albo_eigenquantities(
    #     angle=45.0, scale=33.0
    # )

    albo_evals, albo_evecs, mass = pca_eigen_albo.get_albo_eigenquantities(
        angles=torch.deg2rad(torch.tensor([45.0, 18.3, 10.0], device=device)),
        scales=torch.tensor([33.0, 60, 5.2], device=device),
    )

    colours = torch.zeros([3, *verts.shape], device=device)

    idxs = torch.randint(0, mesh.vertices.shape[0], (3,))
    # idxs = torch.tensor([3804, 0, 4274], device=device)
    # colours[:, idxs, :] = torch.tensor(
    #     [1.0, 0, 0], dtype=torch.float64
    # ).unsqueeze(0)
    # colours[:, 0, :] = torch.tensor([1.0, 0, 0]).unsqueeze(0)
    colours[torch.arange(3), idxs, :] = torch.tensor(
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=torch.float, device=device
    )

    colours = heat_diffusion(
        colours,
        mass,
        albo_evals,
        albo_evecs,
        torch.tensor([0.001, 0.1, 0.01], device=device),
        at_vertices=True,
    )

    colours = colours.sum(dim=0)

    colours = (colours - colours.min()) / (colours.max() - colours.min())
    colours *= 255
    colours = colours.squeeze().cpu().numpy()

    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colours)
