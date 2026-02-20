from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "_heat_self_from_src",
    "_heat_qk_from_src_query",
]


def _resolve_time_self(time: Tensor, evals_src: Tensor) -> Tensor:
    # evals_src: [S, E]
    if time.ndim == 0:
        return time.view(1, 1)
    if time.ndim != 1:
        raise ValueError(f"time must be scalar or [S], got shape {tuple(time.shape)}")
    if time.shape[0] != evals_src.shape[0]:
        raise ValueError(
            f"time shape mismatch: expected S={evals_src.shape[0]}, got {time.shape[0]}"
        )
    return time.view(-1, 1)


def _resolve_time_qk(time: Tensor, evals_knn: Tensor, src_knn_idx: Tensor) -> Tensor:
    # evals_knn: [Q, K, E], src_knn_idx: [Q, K]
    if time.ndim == 0:
        return time.view(1, 1, 1)
    if time.ndim == 1:
        # Interpret as per-source time [S], gathered to [Q, K].
        # Avoid value-based tensor checks in compiled path.
        gathered = torch.index_select(time, 0, src_knn_idx.reshape(-1))
        return gathered.view(src_knn_idx.shape[0], src_knn_idx.shape[1], 1)
    if time.ndim == 2:
        if time.shape[0] != evals_knn.shape[0] or time.shape[1] != evals_knn.shape[1]:
            raise ValueError(
                "time [Q,K] shape mismatch with evals_knn "
                f"{tuple(evals_knn.shape[:2])}, got {tuple(time.shape)}"
            )
        return time.unsqueeze(-1)
    raise ValueError(
        f"time must be scalar, [S], or [Q,K], got shape {tuple(time.shape)}"
    )


def _heat_self_from_src(evals_src: Tensor, evecs_src: Tensor, time: Tensor) -> Tensor:
    # h_t(s,s) = sum_e exp(-lambda_e t) * phi_e(s)^2
    if evals_src.ndim != 2 or evecs_src.ndim != 2:
        raise ValueError("evals_src and evecs_src must be [S, E]")
    if evals_src.shape != evecs_src.shape:
        raise ValueError(
            f"shape mismatch: evals_src={tuple(evals_src.shape)} "
            f"evecs_src={tuple(evecs_src.shape)}"
        )

    t = _resolve_time_self(time, evals_src)
    decay = torch.exp(-evals_src * t)
    return (decay * evecs_src * evecs_src).sum(dim=-1)


def _heat_qk_from_src_query(
    evals_knn: Tensor,
    evecs_knn: Tensor,
    evecs_src_knn: Tensor,
    time: Tensor,
    src_knn_idx: Tensor,
) -> Tensor:
    # h_t(s,q) = sum_e exp(-lambda_e t) * phi_e(q) * phi_e(s)
    if evals_knn.ndim != 3 or evecs_knn.ndim != 3 or evecs_src_knn.ndim != 3:
        raise ValueError("evals_knn/evecs_knn/evecs_src_knn must be [Q, K, E]")
    if evals_knn.shape != evecs_knn.shape or evals_knn.shape != evecs_src_knn.shape:
        raise ValueError(
            f"shape mismatch: evals_knn={tuple(evals_knn.shape)} "
            f"evecs_knn={tuple(evecs_knn.shape)} "
            f"evecs_src_knn={tuple(evecs_src_knn.shape)}"
        )

    t = _resolve_time_qk(time, evals_knn, src_knn_idx)
    decay = torch.exp(-evals_knn * t)
    return (decay * evecs_knn * evecs_src_knn).sum(dim=-1)
