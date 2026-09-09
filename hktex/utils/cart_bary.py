import torch
import torch.linalg as linalg

from .typing import *

__all__ = [
    "bary_to_cart_coords",
    "cart_to_bary_coords",
]


def bary_to_cart_coords(
    bary_coords: Float[Tensor, "B 3"], verts: Float[Tensor, "B 3 3"]
) -> Float[Tensor, "B 3"]:
    return torch.einsum("ij,ijk->ik", bary_coords, verts)


def cart_to_bary_coords(
    cart_coords: Float[Tensor, "B 3"], verts: Float[Tensor, "B 3 3"]
) -> Float[Tensor, "B 3"]:
    # TODO: Maybe optimize
    v0 = verts[:, 0]
    v1 = verts[:, 1]
    v2 = verts[:, 2]

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = cart_coords - v0

    d00 = linalg.vecdot(v0v1, v0v1)
    d01 = linalg.vecdot(v0v1, v0v2)
    d11 = linalg.vecdot(v0v2, v0v2)
    d20 = linalg.vecdot(v0p, v0v1)
    d21 = linalg.vecdot(v0p, v0v2)

    denom = d00 * d11 - d01 * d01

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    return torch.stack([u, v, w], axis=-1)
