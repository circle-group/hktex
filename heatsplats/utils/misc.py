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
    "compute_face_areas",
    "big_trimesh_pcl",
    "get_rank",
    "get_device",
    "load_module_weights",
    "seed_everything",
    "interpolate_barycentric_attr",
    "interpolate_barycentric_attr_from_trivertidx",
    "normalise_colours",
    "soft_step",
    "rescaled_soft_step",
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
    return compute_face_areas(pos, faces).sum()


def compute_face_areas(pos, faces):
    side_1 = pos[faces[1]] - pos[faces[0]]
    side_2 = pos[faces[2]] - pos[faces[0]]
    return side_1.cross(side_2, dim=1).norm(p=2, dim=1).abs() / 2


def get_all_face_vertices(verts, faces):
    return verts[faces]


def interpolate_barycentric_attr(f, fi, bc, attribute):
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


def interpolate_barycentric_attr_from_trivertidx(tri_vert_idx, bc, attribute):
    """
    Interpolate an attribute stored at each vertex of a mesh across the faces of a
    triangle mesh using barycentric coordinates

    Args:
        tri_vert_idx : [#attribs, 3, 3]-shaped indices of the vertices of the triangles
        bc: [#attribs, 3]-shaped barycentric coordinates for each attribute
        attribute: [#vertices, dim]-shaped attributes at each of the mesh vertices

    Returns:
        [#attribs, dim]-shaped tensor of interpolated attributes.
    """
    return (attribute[tri_vert_idx] * bc[:, :, None]).sum(1)


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


def normalise_colours(colours: Float[Tensor, "B 3"]):
    cmin, cmax = colours.min(), colours.max()
    colours = (colours - cmin) / (cmax - cmin)
    return colours


def soft_step(
    x: Float[Tensor, "G B 1"],
    epsilon: float = 0.8,
    sharpness: Union[float, Float[Tensor, "G"]] = 10.0,
):
    # epsilon sets the threshold location.
    # sharpness controls how abrupt the transition is. Higher = closer to hard threshold
    # This function outputs values in (0, 1)
    if isinstance(sharpness, torch.Tensor):
        sharpness = sharpness.view(-1, 1, 1)
    return torch.sigmoid(sharpness * (x - epsilon))


def rescaled_soft_step(
    x: Tensor,
    epsilon: Union[float, Tensor] = 0.8,
    sharpness: Union[float, Tensor] = 10.0,
):
    """
    A soft step function that maps input [0, 1] to output [0, 1].

    This function is guaranteed to be 0 at x=0 and 1 at x=1.
    """
    if isinstance(sharpness, torch.Tensor):
        sharpness = sharpness.view(-1, 1, 1)
    if isinstance(epsilon, torch.Tensor):
        epsilon = epsilon.view(-1, 1, 1)

    # Calculate the sigmoid values at x, 0, and 1
    y = torch.sigmoid(sharpness * (x - epsilon))
    y0 = torch.sigmoid(sharpness * (0.0 - epsilon))
    y1 = torch.sigmoid(sharpness * (1.0 - epsilon))

    # Rescale the output to be exactly in the [0, 1] range
    # Add a small constant to the denominator to avoid division by zero
    rescaled_y = (y - y0) / (y1 - y0 + 1e-8)

    # Clamp the output to ensure it's strictly within [0, 1] due to potential
    # floating point inaccuracies.
    return torch.clamp(rescaled_y, 0.0, 1.0)


class SoftStep:
    def __init__(self, epsilon=0.8, sharpness=10.0, normlise=False):
        self.epsilon = epsilon
        self.sharpness = sharpness
        if normlise:
            self.normalization_factor = torch.sigmoid(sharpness * (1 - epsilon))
        else:
            self.normalization_factor = 1.0

    def __call__(self, x):
        return soft_step(x, self.epsilon, self.sharpness) / self.normalization_factor
