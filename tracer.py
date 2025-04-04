from abc import abstractmethod
import numpy as np

import potpourri3d as pp3d

from utils.typing import *


class GeodesicTracer:
    def __init__(self):
        pass

    @abstractmethod
    def trace(
        self,
        barycentric_coords: Float[Tensor, "B 3"],
        face_ids: Int[Tensor, "B"],
        tangent_vector: Float[Tensor, "B 3"],
    ) -> tuple[Float[Tensor, "B 3"], Int[Tensor, "B"]]:
        pass


class CPUGeodesicTracer(GeodesicTracer):
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        max_iterations: Optional[int] = None,
    ):
        super().__init__()

        self.V = vertices.shape[0]
        self.F = faces.shape[0]
        assert vertices.shape[1] == 3 and faces.shape[1] == 3

        self.tracer = pp3d.GeodesicTracer(vertices, faces)
        self.max_iterations = max_iterations

    def trace(
        self,
        barycentric_coords: Float[Tensor, "B 3"],
        face_ids: Int[Tensor, "B"],
        tangent_vector: Float[Tensor, "B 3"],
    ) -> tuple[Float[Tensor, "B 3"], Int[Tensor, "B"]]:
        coord_np = barycentric_coords.numpy()
        face_id_np = face_ids.numpy()
        tangent_np = tangent_vector.numpy()

        B = coord_np.shape[0]
        assert face_id_np.shape[0] == B and tangent_np.shape[0] == B

        for i in range(B):
            # TODO
            pass
