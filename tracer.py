from abc import abstractmethod
import trimesh
import numpy as np

import torch
import potpourri3d as pp3d

from utils.typing import *
from utils.geodesics import *


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

        self._mesh = trimesh.Trimesh(vertices, faces)

    def trace(
        self,
        barycentric_coords: Float[Tensor, "B 3"],
        face_ids: Int[Tensor, "B"],
        tangent_vector: Float[Tensor, "B 3"],
    ) -> tuple[Float[Tensor, "B 3"], Int[Tensor, "B"]]:
        coord_np = barycentric_coords.cpu().numpy()
        face_id_np = face_ids.cpu().numpy()
        tangent_np = tangent_vector.cpu().numpy()

        B = coord_np.shape[0]
        assert face_id_np.shape[0] == B and tangent_np.shape[0] == B

        new_coords = []
        for i in range(B):
            # TODO: do the tangent_euclid in torch cuda
            vert_ids = self._mesh.faces[face_id_np[i]]
            vertices = self._mesh.vertices[vert_ids]
            coord_euclid = np.sum(coord_np[i] * (vertices), axis=0).reshape((1, 3))
            tangent_euclid = np.sum(tangent_np[i] * (vertices - coord_euclid), axis=0)
            trace_pts = self.tracer.trace_geodesic_from_face(
                face_id_np[i], coord_np[i], tangent_euclid, self.max_iterations
            )
            new_coords.append(trace_pts[-1, :])
        new_coords = np.stack(new_coords)
        _, _, new_face_ids = trimesh.proximity.closest_point(self._mesh, new_coords)
        new_barycentric_coords = batched_cartesian_to_barycentric_coordinates(
            new_coords,
            get_all_face_vertices(self._mesh.vertices, self._mesh.faces)[new_face_ids],
        )

        new_barycentric_coords = barycentric_coords.new_tensor(new_barycentric_coords)
        new_face_ids = face_ids.new_tensor(new_face_ids)
        return new_barycentric_coords, new_face_ids
