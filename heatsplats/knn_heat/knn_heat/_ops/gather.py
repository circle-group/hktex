from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

__all__ = [
    "_gather_m_knn",
    "_repeat_query_geom",
    "_gather_evals_flat",
    "_evecs_full",
    "_evecs_stream_vertices",
    "_evecs_stream_r_i",
]


def _gather_m_knn(
    src_knn_idx: Tensor,
    m_neigh_idx: Tensor,
    m_neigh_w: Tensor,
) -> Tuple[Tensor, Tensor]:
    # src_knn_idx: [Q, K], m_neigh_idx: [S, 4], m_neigh_w: [S, 4]
    # returns: m_idx [N, 4], m_w [N, 4]
    src_flat = src_knn_idx.reshape(-1)
    m_idx = m_neigh_idx.index_select(0, src_flat)
    m_w = m_neigh_w.index_select(0, src_flat)
    return m_idx, m_w


def _repeat_query_geom(
    src_knn_idx: Tensor,
    vert_idx: Tensor,
    bary_coords: Tensor,
) -> Tuple[Tensor, Tensor]:
    # vert_idx: [Q, 3], bary_coords: [Q, 3]
    # returns: v [N, 3], b [N, 3]
    q, k = src_knn_idx.shape
    q_idx = torch.arange(q, device=src_knn_idx.device).repeat_interleave(k)
    v = vert_idx.index_select(0, q_idx)
    b = bary_coords.index_select(0, q_idx)
    return v, b


def _gather_evals_flat(m_idx: Tensor, m_w: Tensor, evals: Tensor) -> Tensor:
    # m_idx: [N, 4], m_w: [N, 4], evals: [M, E]
    # returns: [N, E]
    gathered = evals[m_idx]  # [N, 4, E]
    return (gathered * m_w.unsqueeze(-1)).sum(dim=1)


def _evecs_full(
    m_idx: Tensor, m_w: Tensor, v: Tensor, b: Tensor, evecs: Tensor
) -> Tensor:
    # m_idx: [N, 4], v: [N, 3], b: [N, 3], evecs: [M, V, E]
    # returns: [N, E]
    n = m_idx.shape[0]
    gathered = evecs[m_idx.unsqueeze(-1), v.unsqueeze(1)]  # [N, 4, 3, E]
    evecs_at_p = (gathered * b.view(n, 1, 3, 1)).sum(dim=2)  # [N, 4, E]
    return (evecs_at_p * m_w.unsqueeze(-1)).sum(dim=1)  # [N, E]


def _evecs_stream_vertices(
    m_idx: Tensor, m_w: Tensor, v: Tensor, b: Tensor, evecs: Tensor
) -> Tensor:
    # m_idx: [N, 4], v: [N, 3], b: [N, 3], evecs: [M, V, E]
    # returns: [N, E]
    n = m_idx.shape[0]
    e_dim = evecs.shape[-1]
    evecs_at_p = evecs.new_zeros((n, 4, e_dim))

    for i in range(3):
        v_i = v[:, i].unsqueeze(1).expand(-1, m_idx.shape[1])
        evecs_at_p = evecs_at_p + evecs[m_idx, v_i] * b[:, i].view(n, 1, 1)

    return (evecs_at_p * m_w.unsqueeze(-1)).sum(dim=1)


def _evecs_stream_r_i(
    m_idx: Tensor, m_w: Tensor, v: Tensor, b: Tensor, evecs: Tensor
) -> Tensor:
    # m_idx: [N, 4], v: [N, 3], b: [N, 3], evecs: [M, V, E]
    # returns: [N, E]
    n = m_idx.shape[0]
    e_dim = evecs.shape[-1]
    out = evecs.new_zeros((n, e_dim))

    for r in range(4):
        m_r = m_idx[:, r]
        w_r = m_w[:, r].view(n, 1)
        for i in range(3):
            out = out + evecs[m_r, v[:, i]] * (w_r * b[:, i].view(n, 1))

    return out
