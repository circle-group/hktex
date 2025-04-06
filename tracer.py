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
        coords: Float[Tensor, "B 3"],
        face_ids: Int[Tensor, "B"],
        tangent_vector: Float[Tensor, "B 3"],
        *,
        bary_coords: Optional[Float[Tensor, "B 3"]] = None,
        out_coords: Optional[Float[Tensor, "B 3"]] = None,
        out_face_ids: Optional[Float[Tensor, "B"]] = None,
    ) -> tuple[Float[Tensor, "B 3"], Int[Tensor, "B"]]:
        pass

    def trace_(
        self,
        coords: Float[Tensor, "B 3"],
        face_ids: Int[Tensor, "B"],
        tangent_vector: Float[Tensor, "B 3"],
        *,
        bary_coords: Optional[Float[Tensor, "B 3"]] = None,
    ) -> tuple[Float[Tensor, "B 3"], Int[Tensor, "B"]]:
        return self.trace(
            coords,
            face_ids,
            tangent_vector,
            bary_coords=bary_coords,
            out_coords=coords,
            out_face_ids=face_ids,
        )


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
        self._traces_info = (
            {
                "traces": [[] for _ in range(n_debug_traces)],
                "starts": [[] for _ in range(n_debug_traces)],
            }
            if debug
            else None
        )
        self._n_debug_traces = n_debug_traces

    def trace(
        self,
        coords: Float[Tensor, "B 3"],
        face_ids: Int[Tensor, "B"],
        tangent_vector: Float[Tensor, "B 3"],
        *,
        bary_coords: Optional[Float[Tensor, "B 3"]] = None,
        out_coords: Optional[Float[Tensor, "B 3"]] = None,
        out_face_ids: Optional[Float[Tensor, "B"]] = None,
    ) -> tuple[Float[Tensor, "B 3"], Int[Tensor, "B"]]:
        if bary_coords is None:
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
                self._traces_info["traces"][i].append(trace_pts)
                self._traces_info["starts"][i].append(trace_pts[0, :])

        new_coords = np.stack(new_coords)
        _, _, new_face_ids = trimesh.proximity.closest_point(self._mesh, new_coords)

        if out_coords is None:
            out_coords = coords.new_tensor(new_coords)
        else:
            out_coords.copy_(torch.tensor(new_coords, device="cpu"))

        if out_face_ids is None:
            out_face_ids = face_ids.new_tensor(new_face_ids)
        else:
            out_face_ids.copy_(torch.tensor(new_face_ids, device="cpu"))
        return out_coords, out_face_ids

    @property
    def full_traces_info(self):
        assert self._traces_info is not None, "Tracing info is not enabled."

        # Since trajectories have been separately stored for each heat source as
        # separate lists of arrays now create only one array for each trajectory
        # NB: To represent the trajectories don't need to repeat end of a segment
        # and start of following.
        self._traces_info["traces"] = [
            np.stack([*s, self._traces_info["traces"][i][-1][1, :]])
            for i, s in enumerate(self._traces_info["starts"])
        ]
        self._traces_info["starts"] = [np.stack(s) for s in self._traces_info["starts"]]

        # Since trajectories can be short, identify triangle changes and keep only
        # first and last pairs on same triangle + transition to new triangle
        # Source points will still represent the initial point of every
        # optimisation step
        for i in range(len(self._traces_info["traces"])):
            trace = self._traces_info["traces"][i]
            _, _, face_ids = trimesh.proximity.closest_point(self._mesh, trace)
            new_trace = [trace[0]]
            for j in range(1, len(trace)):
                if face_ids[j] != face_ids[j - 1]:
                    new_trace.append(trace[j - 1])
                    new_trace.append(trace[j])
            new_trace.append(trace[-1])
            self._traces_info["traces"][i] = np.stack(new_trace)
        return self._traces_info

    def reset_traces_info(self):
        if self._traces_info is not None:
            for i in range(len(self._traces_info["traces"])):
                self._traces_info["traces"][i] = []
                self._traces_info["starts"][i] = []
        else:
            raise ValueError("Tracing info is not enabled.")
