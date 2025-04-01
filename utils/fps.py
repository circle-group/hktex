import torch

__all__ = ["farthest_point_sampling", "cross", "dot", "norm2"]


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
