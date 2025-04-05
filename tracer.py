from abc import abstractmethod
import trimesh
import numpy as np

import torch
import potpourri3d as pp3d

import utils
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
        vertices: torch.Tensor,
        faces: torch.Tensor,
        max_iterations: Optional[int] = None,
        debug: bool = False,
        n_debug_traces: int = 10,
    ):
        super().__init__()

        self.V = vertices.shape[0]
        self.F = faces.shape[0]
        assert vertices.shape[1] == 3 and faces.shape[1] == 3
        self.vertices = vertices
        self.faces = faces
        vertices_np = vertices.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()

        self.tracer = pp3d.GeodesicTracer(vertices_np, faces_np)
        self.max_iterations = max_iterations

        self._mesh = trimesh.Trimesh(vertices_np, faces_np)
        self._traces_info = {"traces": [], "sources": []} if debug else None
        self._n_debug_traces = n_debug_traces

    def trace(
        self,
        coords: Float[Tensor, "B 3"],
        face_ids: Int[Tensor, "B"],
        tangent_vector: Float[Tensor, "B 3"],
    ) -> tuple[Float[Tensor, "B 3"], Int[Tensor, "B"]]:
        # TODO: Take barycentric coords directly instead of euclidian coords
        #       as it is calculated by the code already
        vert_idx = self.faces[face_ids]
        B, T = vert_idx.shape
        vertx = self.vertices[vert_idx.view(B * T)].view(B, T, -1)
        bary_coords = utils.cart_to_bary_coords(coords, vertx)

        bary_coord_np = bary_coords.detach().cpu().numpy()
        face_id_np = face_ids.detach().cpu().numpy()
        tangent_np = tangent_vector.detach().cpu().numpy()

        B = bary_coord_np.shape[0]
        assert face_id_np.shape[0] == B and tangent_np.shape[0] == B

        new_coords = []
        for i in range(B):
            trace_pts = self.tracer.trace_geodesic_from_face(
                face_id_np[i], bary_coord_np[i], tangent_np[i], self.max_iterations
            )
            new_coords.append(trace_pts[-1, :])
            if self._traces_info is not None and i < self._n_debug_traces:
                self._traces_info["traces"].append(trace_pts)
                self._traces_info["sources"].append(trace_pts[0, :])

        new_coords = np.stack(new_coords)
        _, _, new_face_ids = trimesh.proximity.closest_point(self._mesh, new_coords)
        # new_barycentric_coords = batched_cartesian_to_barycentric_coordinates(
        #     new_coords,
        #     get_all_face_vertices(self._mesh.vertices, self._mesh.faces)[new_face_ids],
        #  )

        # TODO: Have an inplace version that writes these directly to previous tensors
        new_coords = coords.new_tensor(new_coords)
        new_face_ids = face_ids.new_tensor(new_face_ids)
        return new_coords, new_face_ids

    @property
    def full_traces_info(self):
        assert self._traces_info is not None, "Tracing info is not enabled."
        return self._traces_info

    def reset_traces_info(self):
        if self._traces_info is not None:
            self._traces_info["traces"] = []
            self._traces_info["sources"] = []
        else:
            raise ValueError("Tracing info is not enabled.")
