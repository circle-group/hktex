from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FaissGpuIndexConfig:
    metric: Literal["l2", "ip"] = "l2"
    use_float16: bool = False
    compile_distances: bool = True
    distance_impl: Literal["naive", "bmm"] = "bmm"
    build_db_norm: bool = False


@dataclass(frozen=True)
class SpectralKnnGatherConfig:
    select_impl: Literal["knn4", "bilinear4"] = "bilinear4"
    compile_select: bool = True

    gather_impl: Literal["full", "stream_vertices", "stream_r_i"] = "stream_vertices"
    compile_gather: bool = True


@dataclass(frozen=True)
class SpectralKnnHeatConfig:
    compile_heat: bool = True
    eps: float = 1e-8


@dataclass(frozen=True)
class KnnPostDiffWeightConfig:
    kernel: Literal["inverse", "gaussian"] = "inverse"
    compile_kernel: bool = True
    normalize: bool = True
    eps: float = 1e-8
    gaussian_std: float = 1.0
