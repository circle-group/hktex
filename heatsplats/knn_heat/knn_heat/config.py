from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# str instead of Literal for easier integration with omegaconf


@dataclass  # (frozen=True)
class FaissGpuIndexConfig:
    metric: str = "l2"  # Literal["l2", "ip"]
    use_float16: bool = False
    compile_distances: bool = False
    distance_impl: str = "bmm"  # Literal["naive", "bmm"]
    build_db_norm: bool = False


@dataclass  # (frozen=True)
class SpectralKnnGatherConfig:
    select_impl: str = "bilinear4"  # Literal["knn4", "bilinear4"]
    compile_select: bool = False

    gather_impl: str = (
        "stream_vertices"  # Literal["full", "stream_vertices", "stream_r_i"]
    )
    compile_gather: bool = False


@dataclass  # (frozen=True)
class SpectralKnnHeatConfig:
    compile_heat: bool = False
    eps: float = 1e-8


@dataclass  # (frozen=True)
class KnnPostDiffWeightConfig:
    kernel: str = "inverse"  # Literal["inverse", "gaussian"]
    compile_kernel: bool = False
    normalize: bool = True
    eps: float = 1e-8
    gaussian_std: float = 1.0
