import os
import random
import re
import numpy as np
import scipy.sparse

import torch
import trimesh

import heatsplats
from .typing import *

__all__ = [
    "to_np",
    "sparse_torch_to_np",
    "stiefel_projx",
    "compute_tot_area",
    "big_trimesh_pcl",
    "get_rank",
    "get_device",
    "load_module_weights",
    "seed_everything",
    "interpolate_barycentric_coords",
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


def get_all_face_vertices(verts, faces):
    return verts[faces]


def interpolate_barycentric_coords(f, fi, bc, attribute):
    """
    Interpolate an attribute stored at each vertex of a mesh across the faces of a
    triangle mesh using barycentric coordinates

    Args:
        f : [#faces, 3]-shaped mesh faces (indexing into some vertex array).
        fi: [#attribs,)-shaped]indexes into f indicating which face each attribute lies
        bc: [#attribs, 3]-shaped barycentric coordinates for each attribute
        attribute: [#vertices, dim]-shaped attributes at each of the mesh vertices

    Returns:
        [#attribs, dim]-shaped tensor of interpolated attributes.
    """
    return (attribute[f[fi]] * bc[:, :, None]).sum(1)


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


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def load_module_weights(
    path, module_name=None, ignore_modules=None, map_location=None
) -> Tuple[dict, int, int]:
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        map_location = get_device()

    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["state_dict"]
    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any(
                [k.startswith(ignore_module + ".") for ignore_module in ignore_modules]
            )
            if ignore:
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            state_dict_to_load[m.group(1)] = v

    return state_dict_to_load, ckpt["epoch"], ckpt["global_step"]


max_seed_value = 4294967295  # 2^32 - 1 (uint32)
min_seed_value = 0


def seed_everything(seed: int, verbose: bool = True) -> int:
    if not (min_seed_value <= seed <= max_seed_value):
        heatsplats.warn(
            f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = 0

    if verbose:
        heatsplats.info(f"Seed set to {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return seed
