from __future__ import annotations

import torch

__all__ = [
    "_knn_distances_l2_sq",
    "_knn_distances_ip",
    "_knn_distances_l2_sq_bmm",
    "_knn_distances_ip_bmm",
    "_knn_distances_l2_sq_bmm_with_db_norm",
]


def _knn_distances_l2_sq(
    queries: torch.Tensor, database: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # queries: [M, D], database: [N, D], indices: [M, K]
    # returns: [M, K] squared L2 distances
    neighbors = database[indices]  # [M, K, D]
    diff = queries.unsqueeze(1) - neighbors
    return (diff * diff).sum(dim=-1)


def _knn_distances_ip(
    queries: torch.Tensor, database: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # queries: [M, D], database: [N, D], indices: [M, K]
    # returns: [M, K] inner products
    neighbors = database[indices]  # [M, K, D]
    return (queries.unsqueeze(1) * neighbors).sum(dim=-1)


def _knn_distances_l2_sq_bmm(
    queries: torch.Tensor, database: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # queries: [M, D], database: [N, D], indices: [M, K]
    # returns: [M, K] squared L2 distances
    m, d = queries.shape
    k = indices.shape[1]

    flat = indices.reshape(-1)
    neighbors = database.index_select(0, flat).view(m, k, d)

    q_norm = (queries * queries).sum(dim=-1, keepdim=True)
    x_norm = (neighbors * neighbors).sum(dim=-1)
    qx = torch.bmm(neighbors, queries.unsqueeze(-1)).squeeze(-1)

    return q_norm + x_norm - 2.0 * qx


def _knn_distances_ip_bmm(
    queries: torch.Tensor, database: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # queries: [M, D], database: [N, D], indices: [M, K]
    # returns: [M, K] inner products
    m, d = queries.shape
    k = indices.shape[1]

    flat = indices.reshape(-1)
    neighbors = database.index_select(0, flat).view(m, k, d)

    return torch.bmm(neighbors, queries.unsqueeze(-1)).squeeze(-1)


def _knn_distances_l2_sq_bmm_with_db_norm(
    queries: torch.Tensor,
    database: torch.Tensor,
    indices: torch.Tensor,
    db_norm: torch.Tensor,
) -> torch.Tensor:
    # queries: [M, D], database: [N, D], indices: [M, K]
    # returns: [M, K] squared L2 distances
    m, d = queries.shape
    k = indices.shape[1]

    flat = indices.reshape(-1)
    neighbors = database.index_select(0, flat).view(m, k, d)

    q_norm = (queries * queries).sum(dim=-1, keepdim=True)
    x_norm = db_norm.index_select(0, flat).view(m, k)
    qx = torch.bmm(neighbors, queries.unsqueeze(-1)).squeeze(-1)

    return q_norm + x_norm - 2.0 * qx
