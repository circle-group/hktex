from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "_inverse_weighting_kernel",
    "_gaussian_weighting_kernel",
]


def _inverse_weighting_kernel(
    pts2knn_dist: Tensor,
    *,
    eps: float = 1e-16,
    normalize: bool = True,
) -> Tensor:
    # pts2knn_dist: [Q, K] (or any [..., K] layout)
    if pts2knn_dist.ndim < 2:
        raise ValueError(
            f"pts2knn_dist must have at least 2 dims [..., K], got {tuple(pts2knn_dist.shape)}"
        )

    weights = 1.0 / torch.clamp(pts2knn_dist, min=eps)
    if normalize:
        weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=eps)
    return weights


def _gaussian_weighting_kernel(
    pts2knn_dist: Tensor,
    *,
    std: float,
    eps: float = 1e-16,
    normalize: bool = True,
) -> Tensor:
    # pts2knn_dist: [Q, K] (or any [..., K] layout)
    if pts2knn_dist.ndim < 2:
        raise ValueError(
            f"pts2knn_dist must have at least 2 dims [..., K], got {tuple(pts2knn_dist.shape)}"
        )
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}")

    weights = torch.exp(-(pts2knn_dist * pts2knn_dist) / (2.0 * std * std))
    if normalize:
        weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=eps)
    return weights
