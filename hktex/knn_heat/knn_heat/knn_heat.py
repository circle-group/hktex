from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .config import SpectralKnnHeatConfig
from . import _ops

__all__ = ["SpectralKnnHeat"]


@dataclass(frozen=True)
class _SourceHeatCache:
    # h_t(s,s), used as normalization denominator for all query-time gathers.
    src_self_heat: Float[Tensor, "S"]
    evecs_src: Float[Tensor, "S E"]


class SpectralKnnHeat:
    """
    Two-stage heat diffusion over KNN-selected source-query pairs.

    Typical training pattern:
      1) build(...) once after source gather to compute/cache h_t(s,s)
      2) query(...) inside the query loop for h_t(s,q), optionally normalized
         by the cached per-source self heat
    """

    def __init__(self, config: SpectralKnnHeatConfig | None = None) -> None:
        self.config = config or SpectralKnnHeatConfig()
        self._heat_self_fn = self._make_heat_self_fn()
        self._heat_qk_fn = self._make_heat_qk_fn()
        self._cache: Optional[_SourceHeatCache] = None
        self._time_built: Optional[Tensor] = None

    def _make_heat_self_fn(self) -> Callable[..., Tensor]:
        fn = _ops._heat_self_from_src
        if self.config.compile_heat and hasattr(torch, "compile"):
            return torch.compile(fn, dynamic=True)
        return fn

    def _make_heat_qk_fn(self) -> Callable[..., Tensor]:
        fn = _ops._heat_qk_from_src_query
        if self.config.compile_heat and hasattr(torch, "compile"):
            return torch.compile(fn, dynamic=True)
        return fn

    def _build_source_cache(
        self,
        evals_src: Float[Tensor, "S E"],
        evecs_src: Float[Tensor, "S E"],
        time: Float[Tensor, ""] | Float[Tensor, "S"],
    ) -> _SourceHeatCache:
        """
        Build per-source reference normalizer h_t(s,s).
        """
        assert evals_src.ndim == 2 and evecs_src.ndim == 2
        assert evals_src.shape == evecs_src.shape

        src_self_heat = self._heat_self_fn(evals_src, evecs_src, time)
        return _SourceHeatCache(
            src_self_heat=src_self_heat,
            evecs_src=evecs_src,
        )

    def reset(self) -> None:
        self._cache = None
        self._time_built = None

    def build(
        self,
        evals_src: Float[Tensor, "S E"],
        evecs_src: Float[Tensor, "S E"],
        time: Float[Tensor, ""] | Float[Tensor, "S"],
    ) -> None:
        """
        Build (or rebuild) the internal source cache for repeated query calls.
        """
        cache = self._build_source_cache(evals_src, evecs_src, time)
        self._cache = cache
        self._time_built = time

    def query(
        self,
        src_knn_idx: Int64[Tensor, "Q K"],
        evals_knn: Float[Tensor, "Q K E"],
        evecs_knn: Float[Tensor, "Q K E"],
        *,
        evecs_src_knn: Optional[Float[Tensor, "Q K E"]] = None,
        time: Optional[
            Float[Tensor, ""] | Float[Tensor, "Q K"] | Float[Tensor, "S"]
        ] = None,
        weights_post_diff: Optional[Float[Tensor, "Q K"] | Float[Tensor, "Q"]] = None,
        normalize: bool = True,
        src_self_heat_knn: Optional[Float[Tensor, "Q K"]] = None,
    ) -> Tuple[Float[Tensor, "Q K"], Optional[Float[Tensor, "Q K"]]]:
        """
        Query heat values using the built source cache.

        Args:
            src_knn_idx: [Q, K] source indices per query
            evals_knn: [Q, K, E] source-conditioned evals already gathered for knn pairs
            evecs_knn: [Q, K, E] source-conditioned query evecs already gathered for knn pairs
            evecs_src_knn: optional [Q, K, E] pre-gathered source evecs for each knn pair.
                If omitted, gathered from cached source evecs via src_knn_idx.
            time: optional override for diffusion time; if None, use time from build()
            weights_post_diff: optional post-diffusion weights for query-side values.
                Can be [Q,K] (per pair) or [Q] (broadcast across K). Applied before normalization.
            normalize: if True, return normalized heat using cached h_t(s,s)
            src_self_heat_knn: optional [Q, K] pre-gathered normalization denominator.
                If omitted and normalize=True, gathered from cached src_self_heat via src_knn_idx.
        """
        if self._cache is None:
            raise RuntimeError(
                "Cache is not built. Call build(evals_src, evecs_src, time) first."
            )

        t = time if time is not None else self._time_built
        if t is None:
            raise RuntimeError(
                "No diffusion time available. Pass time=... or call build(...) first."
            )

        e_src_knn = evecs_src_knn
        if e_src_knn is None:
            e_src_knn = self._cache.evecs_src[src_knn_idx]

        norm_knn = src_self_heat_knn
        if normalize and norm_knn is None:
            norm_knn = self._cache.src_self_heat[src_knn_idx]

        return self._diffuse_knn_queries(
            src_knn_idx=src_knn_idx,
            evals_knn=evals_knn,
            evecs_knn=evecs_knn,
            evecs_src_knn=e_src_knn,
            time=t,
            weights_post_diff=weights_post_diff,
            src_self_heat_knn=norm_knn if normalize else None,
        )

    def _diffuse_knn_queries(
        self,
        src_knn_idx: Int64[Tensor, "Q K"],
        evals_knn: Float[Tensor, "Q K E"],
        evecs_knn: Float[Tensor, "Q K E"],
        evecs_src_knn: Float[Tensor, "Q K E"],
        time: Float[Tensor, ""] | Float[Tensor, "Q K"] | Float[Tensor, "S"],
        weights_post_diff: Optional[Float[Tensor, "Q K"] | Float[Tensor, "Q"]] = None,
        src_self_heat_knn: Optional[Float[Tensor, "Q K"]] = None,
    ) -> Tuple[Float[Tensor, "Q K"], Optional[Float[Tensor, "Q K"]]]:
        """
        Compute h_t(s,q) over KNN pairs and optionally normalize by h_t(s,s).

        Returns:
            heat_qk: [Q, K]
            heat_qk_norm: [Q, K] or None
        """
        assert src_knn_idx.ndim == 2
        assert evals_knn.ndim == 3 and evecs_knn.ndim == 3
        assert evals_knn.shape == evecs_knn.shape
        assert evecs_src_knn.ndim == 3 and evecs_src_knn.shape == evals_knn.shape
        heat_qk = self._heat_qk_fn(
            evals_knn,
            evecs_knn,
            evecs_src_knn,
            time,
            src_knn_idx,
        )
        if weights_post_diff is not None:
            if weights_post_diff.ndim == 1:
                assert weights_post_diff.shape[0] == src_knn_idx.shape[0]
                heat_qk = heat_qk * weights_post_diff.unsqueeze(1)
            else:
                assert (
                    weights_post_diff.ndim == 2
                    and weights_post_diff.shape == src_knn_idx.shape
                )
                heat_qk = heat_qk * weights_post_diff

        heat_qk_norm = None
        if src_self_heat_knn is not None:
            assert src_self_heat_knn.ndim == 2 and src_self_heat_knn.shape == src_knn_idx.shape
            heat_qk_norm = heat_qk / (src_self_heat_knn + self.config.eps)

        return heat_qk, heat_qk_norm
