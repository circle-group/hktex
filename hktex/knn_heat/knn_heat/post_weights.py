from __future__ import annotations

from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from .config import KnnPostDiffWeightConfig
from . import _ops

__all__ = [
    "inverse_weighting_kernel",
    "gaussian_weighting_kernel",
    "KnnPostDiffWeight",
]


def inverse_weighting_kernel(
    pts2knn_dist: Float[Tensor, "Q K"],
    *,
    eps: float = 1e-16,
    normalize: bool = True,
) -> Float[Tensor, "Q K"]:
    """
    Inverse-distance kernel for query-side post-diffusion weights.

    Notes:
    - Returns [Q, K] for the current pipeline (no extra self/source column).
    - Use with SpectralKnnHeat.query(..., weights_post_diff=...).
    """
    return _ops._inverse_weighting_kernel(
        pts2knn_dist,
        eps=eps,
        normalize=normalize,
    )


def gaussian_weighting_kernel(
    pts2knn_dist: Float[Tensor, "Q K"],
    *,
    std: float,
    eps: float = 1e-16,
    normalize: bool = True,
) -> Float[Tensor, "Q K"]:
    """
    Gaussian kernel for query-side post-diffusion weights.

    Notes:
    - Returns [Q, K] for the current pipeline (no extra self/source column).
    - Use with SpectralKnnHeat.query(..., weights_post_diff=...).
    """
    return _ops._gaussian_weighting_kernel(
        pts2knn_dist,
        std=std,
        eps=eps,
        normalize=normalize,
    )


class KnnPostDiffWeight:
    """
    Compute weights_post_diff from differentiable KNN distances.

    Typical usage:
      1) distances = index.knn_distances(queries, indices)   # [Q, K], differentiable
      2) weights = weighter.compute(distances)               # [Q, K]
      3) heat.query(..., weights_post_diff=weights)
    """

    def __init__(self, config: KnnPostDiffWeightConfig | None = None) -> None:
        self.config = config or KnnPostDiffWeightConfig()
        if self.config.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.config.eps}")
        if self.config.gaussian_std <= 0.0:
            raise ValueError(
                f"gaussian_std must be positive, got {self.config.gaussian_std}"
            )
        self._kernel_fn = self._make_kernel_fn()

    def _make_kernel_fn(self) -> Callable[[Tensor], Tensor]:
        if self.config.kernel == "inverse":

            def fn(d: Tensor) -> Tensor:
                return _ops._inverse_weighting_kernel(
                    d,
                    eps=self.config.eps,
                    normalize=self.config.normalize,
                )

        elif self.config.kernel == "gaussian":

            def fn(d: Tensor) -> Tensor:
                return _ops._gaussian_weighting_kernel(
                    d,
                    std=self.config.gaussian_std,
                    eps=self.config.eps,
                    normalize=self.config.normalize,
                )

        else:
            raise ValueError(f"Unsupported kernel: {self.config.kernel}")

        if self.config.compile_kernel and hasattr(torch, "compile"):
            return torch.compile(fn, dynamic=True)
        return fn

    def compute(self, pts2knn_dist: Float[Tensor, "Q K"]) -> Float[Tensor, "Q K"]:
        # faiss_knn returns squared L2 distances for metric="l2"
        pts2knn_dist = torch.sqrt(torch.clamp(pts2knn_dist, min=0.0) + self.config.eps)
        return self._kernel_fn(pts2knn_dist)

    def __call__(self, pts2knn_dist: Float[Tensor, "Q K"]) -> Float[Tensor, "Q K"]:
        return self.compute(pts2knn_dist)

    def compute_from_faiss(
        self,
        index: object,
        queries: Tensor,
        indices: Tensor,
    ) -> Float[Tensor, "Q K"]:
        """
        Convenience wrapper around a FaissGpuFlatIndex-like object exposing
        knn_distances(queries, indices) -> [Q, K].
        """
        if not hasattr(index, "knn_distances"):
            raise TypeError(
                "index must provide a knn_distances(queries, indices) method"
            )
        dists = index.knn_distances(queries, indices)
        return self.compute(dists)
