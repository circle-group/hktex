import torch

from hktex.utils import compute_face_areas
from .typing import *

__all__ = [
    "farthest_point_sampling",
    "uniform_sampling",
    "uniform_sample_triangle",
    "cross",
    "dot",
    "norm2",
]


def cross(vec_1: torch.Tensor, vec_2: torch.Tensor) -> torch.Tensor:
    return torch.cross(vec_1, vec_2, dim=-1)


def dot(vec_1: torch.Tensor, vec_2: torch.Tensor) -> torch.Tensor:
    return torch.sum(vec_1 * vec_2, dim=-1)


def norm2(x: torch.Tensor) -> torch.Tensor:
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape)
    after norm along last dimension
    """
    return dot(x, x)


def farthest_point_sampling(points: torch.Tensor, n_sample: int) -> torch.Tensor:
    # Torch in, torch out. Returns a |V| mask with n_sample elements set to true

    N = points.shape[0]
    if n_sample > N:
        raise ValueError("not enough points to sample")

    chosen_mask = torch.zeros(N, dtype=torch.bool, device=points.device)
    min_dists = torch.ones(N, dtype=points.dtype, device=points.device) * float("inf")

    # pick the centermost first point
    # points = normalize_positions(points)  # they should be already centered
    i = torch.min(norm2(points), dim=0).indices
    chosen_mask[i] = True

    for _ in range(n_sample - 1):
        # update distance
        dists = norm2(points[i, :].unsqueeze(0) - points)
        min_dists = torch.minimum(dists, min_dists)

        # take the farthest
        i = torch.max(min_dists, dim=0).indices.item()
        chosen_mask[i] = True

    return chosen_mask


def uniform_sample_triangle(uniform_samples: Float[Tensor, "B 2"]):
    su0 = uniform_samples[:, 0].sqrt()
    b0 = 1 - su0
    b1 = uniform_samples[:, 1] * su0
    return torch.stack((b0, b1, 1 - b0 - b1), dim=1)


def uniform_sampling(
    verts: torch.Tensor, faces: torch.Tensor, n_samples: int
) -> torch.Tensor:
    prob = compute_face_areas(verts, faces.T)
    prob = prob / prob.sum()
    face_ids = torch.multinomial(prob, n_samples, replacement=True)

    bary = uniform_sample_triangle(torch.rand((n_samples, 2), device=verts.device))

    return face_ids, bary
