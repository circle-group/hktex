from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .config import SpectralKnnGatherConfig
from . import _ops

__all__ = ["SpectralKnnGather"]


class SpectralKnnGather:
    """
    Gather and interpolate spectral quantities for sources and KNN-selected queries.

    Typical training pattern:
      1) gather_src(...) once to prepare per-source evals/evecs (for precomputed normalizers)
      2) inside the query minibatch loop, call gather_queries(...) to get per-(query,knn) evals/evecs
         and (optionally) gather per-source normalizers for those KNN indices

    Notes:
    - Discrete indices are used for routing only: src_knn_idx, m_neigh_idx, vert_idx
    - Gradients flow through floating inputs: m_neigh_w and bary_coords
    - evals and evecs are treated as constants (no gradients required)
    """

    def __init__(self, config: SpectralKnnGatherConfig | None = None) -> None:
        self.config = config or SpectralKnnGatherConfig()
        self._select_fn = self._make_select_fn()
        self._evecs_fn = self._make_evecs_fn()

    def _make_select_fn(self) -> Callable[..., Tuple[Tensor, Tensor]]:
        if self.config.select_impl == "knn4":
            fn = _ops._select_grid_knn4_from_axes
        elif self.config.select_impl == "bilinear4":
            fn = _ops._select_grid_cell4_bilinear
        else:
            raise ValueError(f"Unsupported select_impl: {self.config.select_impl}")

        if self.config.compile_select and hasattr(torch, "compile"):
            return torch.compile(fn, dynamic=True)

        return fn

    def _make_evecs_fn(self) -> Callable[..., Tensor]:
        if self.config.gather_impl == "full":
            fn = _ops._evecs_full
        elif self.config.gather_impl == "stream_vertices":
            fn = _ops._evecs_stream_vertices
        elif self.config.gather_impl == "stream_r_i":
            fn = _ops._evecs_stream_r_i
        else:
            raise ValueError(f"Unsupported gather_impl: {self.config.gather_impl}")

        if self.config.compile_gather and hasattr(torch, "compile"):
            return torch.compile(fn, dynamic=True)

        return fn

    def select_grid(
        self,
        rot_grid: Float[Tensor, "R"],
        scale_grid: Float[Tensor, "A"],
        angles: Float[Tensor, "S"],
        scales: Float[Tensor, "S"],
        *,
        abs_sin: bool = True,
        scale_map: Literal["log1p", "identity"] = "log1p",
        grid_layout: Literal["rot_major", "scale_major"] = "rot_major",
        eps: float = 1e-10,
        clamp_unit: bool = True,
    ) -> Tuple[Int64[Tensor, "S 4"], Float[Tensor, "S 4"]]:
        """
        Select per-source grid neighbors and weights over the rotation x anisotropy grid.

        Returns:
            m_neigh_idx: [S, 4]
            m_neigh_w:   [S, 4]
        """
        assert rot_grid.ndim == 1 and rot_grid.shape[0] >= 2
        assert scale_grid.ndim == 1 and scale_grid.shape[0] >= 2
        assert angles.ndim == 1 and scales.ndim == 1
        assert angles.shape[0] == scales.shape[0]

        idx, w = self._select_fn(
            rot_grid,
            scale_grid,
            angles,
            scales,
            abs_sin=abs_sin,
            scale_map=scale_map,
            grid_layout=grid_layout,
            eps=eps,
            clamp_unit=clamp_unit,
        )

        if idx.dtype != torch.int64:
            idx = idx.to(torch.int64)

        return idx, w

    def gather_src(
        self,
        m_neigh_idx: Int64[Tensor, "S 4"],
        m_neigh_w: Float[Tensor, "S 4"],
        src_vert_idx: Int64[Tensor, "S 3"],
        src_bary_coords: Float[Tensor, "S 3"],
        evals: Float[Tensor, "M E"],
        evecs: Float[Tensor, "M V E"],
    ) -> Tuple[Float[Tensor, "S E"], Float[Tensor, "S E"]]:
        """
        Compute per-source interpolated evals/evecs at the source points.

        Returns:
            evals_src: [S, E]
            evecs_src: [S, E]
        """
        assert (
            m_neigh_idx.ndim == 2
            and m_neigh_idx.shape[1] == 4
            and m_neigh_w.shape == m_neigh_idx.shape
        )
        assert (
            src_vert_idx.ndim == 2
            and src_vert_idx.shape[1] == 3
            and src_bary_coords.shape == src_vert_idx.shape
        )
        assert evals.ndim == 2 and evecs.ndim == 3
        assert evecs.shape[0] == evals.shape[0] and evecs.shape[2] == evals.shape[1]

        evals_src = _ops._gather_evals_flat(m_neigh_idx, m_neigh_w, evals)
        evecs_src = self._evecs_fn(
            m_neigh_idx, m_neigh_w, src_vert_idx, src_bary_coords, evecs
        )
        return evals_src, evecs_src

    def gather_queries(
        self,
        src_knn_idx: Int64[Tensor, "Q K"],
        m_neigh_idx: Int64[Tensor, "S 4"],
        m_neigh_w: Float[Tensor, "S 4"],
        vert_idx: Int64[Tensor, "Q 3"],
        bary_coords: Float[Tensor, "Q 3"],
        evals: Float[Tensor, "M E"],
        evecs: Float[Tensor, "M V E"],
        src_norm: Optional[Float[Tensor, "S"]] = None,
    ) -> Tuple[
        Float[Tensor, "Q K E"],
        Float[Tensor, "Q K E"],
        Optional[Float[Tensor, "Q K"]],
    ]:
        """
        Compute interpolated evals/evecs for KNN-selected sources at query points.

        Returns:
            evals_knn: [Q, K, E]
            evecs_knn: [Q, K, E]
            norm_knn:  [Q, K] or None
        """
        assert (
            m_neigh_idx.ndim == 2
            and m_neigh_idx.shape[1] == 4
            and m_neigh_w.shape == m_neigh_idx.shape
        )
        assert (
            vert_idx.ndim == 2
            and vert_idx.shape[1] == 3
            and bary_coords.shape == vert_idx.shape
        )
        assert (
            evals.ndim == 2
            and evecs.ndim == 3
            and evecs.shape[0] == evals.shape[0]
            and evecs.shape[2] == evals.shape[1]
        )

        q, k = src_knn_idx.shape
        m_idx, m_w = _ops._gather_m_knn(src_knn_idx, m_neigh_idx, m_neigh_w)
        v, b = _ops._repeat_query_geom(src_knn_idx, vert_idx, bary_coords)

        evals_flat = _ops._gather_evals_flat(m_idx, m_w, evals)  # [Q*K, E]
        evecs_flat = self._evecs_fn(m_idx, m_w, v, b, evecs)  # [Q*K, E]

        evals_knn = evals_flat.view(q, k, -1)
        evecs_knn = evecs_flat.view(q, k, -1)

        norm_knn = None
        if src_norm is not None:
            norm_knn = src_norm[src_knn_idx]

        return evals_knn, evecs_knn, norm_knn
