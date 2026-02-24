from typing import TypedDict
from heatsplats.utils.typing import *


class PointsInfo(TypedDict):
    iso_evecs: Float[Tensor, "P K"] | None
    albo_evals: Float[Tensor, "G K"]
    albo_evecs: Float[Tensor, "G P K"]
    mass: Float[Tensor, "1 P"]


class PointsInfoKNN(PointsInfo):
    weights: Float[Tensor, "P K"] | None
    distances: Float[Tensor, "P K"]
    indices: Int64[Tensor, "P K"]


class KernelInfo(TypedDict):
    vert_idx: Float[Tensor, "G 3"]
    barycentric_coords: Float[Tensor, "G 3"]
    albo_evecs: Float[Tensor, "G K"] | None
    mass: Float[Tensor, "G"] | None
