import torch

from .typing import *

__all__ = ["total_variation_loss", "smoothness_loss"]


def _pair_weights(
    nn_distances: Optional[Tensor],
    weighting: str,
    sigma: float,
    eps: float,
    squared_distances: bool,
) -> Optional[Tensor]:
    if nn_distances is None:
        return None

    if weighting == "none":
        return torch.ones_like(nn_distances)

    if weighting == "gaussian":
        d2 = nn_distances if squared_distances else nn_distances.pow(2)
        sigma2 = max(float(sigma), eps) ** 2
        return torch.exp(-d2 / (2.0 * sigma2))

    if weighting == "inverse_distance":
        if squared_distances:
            return torch.rsqrt(nn_distances + eps)
        return 1.0 / (nn_distances + eps)

    raise ValueError(f"Unknown weighting: {weighting}")


def total_variation_loss(
    kernel_colours: Float[Tensor, "G D"],
    nn_indices: Int[Tensor, "G K"],
    nn_distances: Optional[Float[Tensor, "G K"]] = None,
    weighting: str = "gaussian",
    sigma: float = 0.05,
    eps: float = 1e-8,
    squared_distances: bool = True,
    normalise_by_weights: bool = True,
    reduction: str = "mean",
) -> Tensor:
    """
    Graph total-variation (anisotropic TV) on kernel colours.
    """
    center = kernel_colours.unsqueeze(1)  # [G, 1, D]
    neigh = kernel_colours[nn_indices]  # [G, K, D]
    tv_pairs = (neigh - center).abs().sum(dim=-1)  # [G, K]

    w = _pair_weights(nn_distances, weighting, sigma, eps, squared_distances)
    if w is not None:
        tv_pairs = tv_pairs * w
        if normalise_by_weights:
            tv = tv_pairs.sum(dim=1) / (w.sum(dim=1) + eps)
        else:
            tv = tv_pairs.sum(dim=1)
    else:
        tv = tv_pairs.mean(dim=1)

    if reduction == "mean":
        return tv.mean()
    if reduction == "sum":
        return tv.sum()
    if reduction == "none":
        return tv
    raise ValueError(f"Unknown reduction: {reduction}")


def smoothness_loss(
    kernel_colours: Float[Tensor, "G D"],
    nn_indices: Int[Tensor, "G K"],
    nn_distances: Optional[Float[Tensor, "G K"]] = None,
    weighting: str = "gaussian",
    sigma: float = 0.05,
    eps: float = 1e-8,
    squared_distances: bool = True,
    normalise_by_weights: bool = True,
    reduction: str = "mean",
) -> Tensor:
    """
    Graph smoothness (quadratic/Laplacian-style) on kernel colours.
    """
    center = kernel_colours.unsqueeze(1)  # [G, 1, D]
    neigh = kernel_colours[nn_indices]  # [G, K, D]
    sq_pairs = (neigh - center).pow(2).sum(dim=-1)  # [G, K]

    w = _pair_weights(nn_distances, weighting, sigma, eps, squared_distances)
    if w is not None:
        sq_pairs = sq_pairs * w
        if normalise_by_weights:
            smooth = sq_pairs.sum(dim=1) / (w.sum(dim=1) + eps)
        else:
            smooth = sq_pairs.sum(dim=1)
    else:
        smooth = sq_pairs.mean(dim=1)

    if reduction == "mean":
        return smooth.mean()
    if reduction == "sum":
        return smooth.sum()
    if reduction == "none":
        return smooth
    raise ValueError(f"Unknown reduction: {reduction}")
