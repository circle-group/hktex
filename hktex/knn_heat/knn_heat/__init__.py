"""knn_heat: KNN-based spectral gather and heat diffusion utilities."""

from .config import (
    FaissGpuIndexConfig,
    KnnPostDiffWeightConfig,
    SpectralKnnGatherConfig,
    SpectralKnnHeatConfig,
)
from .faiss_knn import FaissGpuFlatIndex
from .knn_gather import SpectralKnnGather
from .knn_heat import SpectralKnnHeat
from .post_weights import (
    KnnPostDiffWeight,
    gaussian_weighting_kernel,
    inverse_weighting_kernel,
)

__version__ = "0.1.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))

__all__ = [
    "__version__",
    "__version_info__",
    "FaissGpuIndexConfig",
    "FaissGpuFlatIndex",
    "KnnPostDiffWeightConfig",
    "KnnPostDiffWeight",
    "inverse_weighting_kernel",
    "gaussian_weighting_kernel",
    "SpectralKnnGatherConfig",
    "SpectralKnnGather",
    "SpectralKnnHeatConfig",
    "SpectralKnnHeat",
]
