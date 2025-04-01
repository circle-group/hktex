import numpy as np
import scipy.sparse

import torch
import trimesh

from .typing import *

__all__ = [
    "to_np",
    "sparse_torch_to_np",
    "stiefel_projx",
    "compute_tot_area",
    "big_trimesh_pcl",
]


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def sparse_torch_to_np(
    mat: torch.sparse.FloatTensor,
) -> scipy.sparse.csc_matrix:
    if len(mat.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(mat.shape))
    mat = mat.coalesce()
    indices = to_np(mat.indices())
    values = to_np(mat.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=mat.shape).tocsc()

    return mat


def stiefel_projx(x: torch.Tensor, driver: Optional[str] = None) -> torch.Tensor:
    assert driver is None or driver in ["gesvd", "gesvda", "gesvdj"]
    U, _, V = torch.linalg.svd(x, full_matrices=False, driver=driver)
    return torch.einsum("...ik,...kj->...ij", U, V)


def compute_tot_area(pos, faces):
    side_1 = pos[faces[1]] - pos[faces[0]]
    side_2 = pos[faces[2]] - pos[faces[0]]
    return side_1.cross(side_2).norm(p=2, dim=1).abs().sum() / 2


def big_trimesh_pcl(points, colours=None, radius=0.015):
    if isinstance(points, torch.Tensor):
        points = to_np(points)
    if isinstance(colours, torch.Tensor):
        colours = to_np(colours)
    pcl = [trimesh.creation.uv_sphere(radius=radius) for i in range(points.shape[0])]

    for i, p in enumerate(pcl):
        p.apply_translation(points[i])
        if colours is not None:
            p.visual.vertex_colors = np.zeros_like(p.vertices) + colours[i]
        else:
            p.visual.vertex_colors = np.zeros_like(p.vertices) + np.array([255, 0, 0])
    return pcl
