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

    index_type: str = "flat"  # "flat" | "ivf_flat" | "cagra"
    ivf_nlist: int = 1024
    ivf_nprobe: int = 16

    # cagra build
    cagra_graph_degree: int = 64
    cagra_intermediate_graph_degree: int = 128
    cagra_build_algo: str = "nn_descent"  # "nn_descent" | "iterative_search" | "ivf_pq"
    cagra_nn_descent_niter: int = 20
    cagra_refine_rate: float = 1.0


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
