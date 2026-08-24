from dataclasses import dataclass, field

from abc import abstractmethod
import trimesh
import numpy as np

import torch
import potpourri3d as pp3d
import digeo
import digeo.ops

from tqdm import tqdm

import heatsplats
from heatsplats.utils import BaseObject
from heatsplats.utils.typing import *

from .mesh import Mesh

__all__ = ["GeodesicTracer", "CPUGeodesicTracer", "GPUGeodesicTracer"]


class GeodesicTracer(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        debug: bool = False

    cfg: Config

    def configure(self, mesh: Mesh):
        super().configure()

        self.mesh = mesh

        self.V = mesh.verts.shape[0]
        self.F = mesh.faces.shape[0]

        self.debug = self.cfg.debug

    @abstractmethod
    def trace(
        self,
        coords: Float[Tensor, "B 3"],
        face_ids: Int[Tensor, "B"],
        tangent_vector: Float[Tensor, "B 3"],
        *,
        bary_coords: Optional[Float[Tensor, "B 3"]] = None,
        transport_vector: Optional[Float[Tensor, "B 3"]] = None,
        out_coords: Optional[Float[Tensor, "B 3"]] = None,
        out_face_ids: Optional[Float[Tensor, "B"]] = None,
        out_transport_vector: Optional[Float[Tensor, "B 3"]] = None,
    ) -> tuple[Float[Tensor, "B 3"], Int[Tensor, "B"]]:
        pass

    def trace_(
        self,
        coords: Float[Tensor, "B 3"],
        face_ids: Int[Tensor, "B"],
        tangent_vector: Float[Tensor, "B 3"],
        *,
        bary_coords: Optional[Float[Tensor, "B 3"]] = None,
        transport_vector: Optional[Float[Tensor, "B 3"]] = None,
    ) -> tuple[Float[Tensor, "B 3"], Int[Tensor, "B"]]:
        return self.trace(
            coords,
            face_ids,
            tangent_vector,
            bary_coords=bary_coords,
            transport_vector=transport_vector,
            out_coords=coords,
            out_face_ids=face_ids,
            out_transport_vector=transport_vector,
        )

    @property
    def full_traces_info(self) -> Optional[Dict[str, Any]]:
        return None

    def reset_traces_info(self):
        pass


@heatsplats.register("modules.cpu-geodesic-tracer")
class CPUGeodesicTracer(GeodesicTracer):
    @dataclass
    class Config(GeodesicTracer.Config):
        max_iterations: Optional[int] = None

        n_debug_traces: int = 10

    cfg: Config

    def configure(self, mesh: Mesh):
        super().configure(mesh)

        vertices_np = self.mesh.verts.detach().cpu().numpy()
        faces_np = self.mesh.faces.detach().cpu().numpy()

        self.tracer = pp3d.GeodesicTracer(vertices_np, faces_np)

        n_debug_traces = self.cfg.n_debug_traces
        self._mesh = trimesh.Trimesh(vertices_np, faces_np, process=False)
        self._traces_info = (
            {
                "traces": [[] for _ in range(n_debug_traces)],
                "starts": [[] for _ in range(n_debug_traces)],
            }
            if self.cfg.debug
            else None
        )

    def trace(
        self,
        coords: Float[Tensor, "B 3"],
        face_ids: Int[Tensor, "B"],
        tangent_vector: Float[Tensor, "B 3"],
        *,
        bary_coords: Optional[Float[Tensor, "B 3"]] = None,
        transport_vector: Optional[Float[Tensor, "B 3"]] = None,
        out_coords: Optional[Float[Tensor, "B 3"]] = None,
        out_face_ids: Optional[Float[Tensor, "B"]] = None,
        out_transport_vector: Optional[Float[Tensor, "B 3"]] = None,
    ) -> tuple[Float[Tensor, "B 3"], Int[Tensor, "B"]]:
        if transport_vector is not None:
            raise RuntimeError(
                "CPUGeodesicTracer doesn't support simultaneous parallel transport"
            )

        if bary_coords is None:
            vert_ids = self.mesh.get_face_vertices(face_ids)
            bary_coords = self.mesh.cartesian_to_barycentric(coords, vert_ids)

        bary_coord_np = bary_coords.detach().cpu().numpy()
        face_id_np = face_ids.detach().cpu().numpy()
        tangent_np = tangent_vector.detach().cpu().numpy()

        B = bary_coord_np.shape[0]
        assert face_id_np.shape[0] == B and tangent_np.shape[0] == B

        new_coords = []
        for i in range(B):
            trace_pts = self.tracer.trace_geodesic_from_face(
                face_id_np[i], bary_coord_np[i], tangent_np[i], self.cfg.max_iterations
            )
            new_coords.append(trace_pts[-1, :])
            if self._traces_info is not None and i < self.cfg.n_debug_traces:
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
    def full_traces_info(self) -> Optional[Dict[str, Any]]:
        assert self._traces_info is not None, "Tracing info is not enabled."

        # Since trajectories have been separately stored for each heat source as
        # separate lists of arrays now create only one array for each trajectory
        traces_info = {}
        traces_info["traces"] = [
            np.concatenate(t, axis=0) for t in self._traces_info["traces"]
        ]
        traces_info["starts"] = [np.stack(s) for s in self._traces_info["starts"]]

        # Since trajectories can be short, identify triangle changes and keep only
        # first and last pairs on same triangle + transition to new triangle
        # Source points will still represent the initial point of every
        # optimisation step
        for i in tqdm(range(len(traces_info["traces"])), "Processing traces"):
            trace = traces_info["traces"][i]
            _, _, face_ids = trimesh.proximity.closest_point(self._mesh, trace)
            new_trace = [trace[0]]
            for j in range(1, len(trace)):
                if face_ids[j] != face_ids[j - 1]:
                    new_trace.append(trace[j - 1])
                    new_trace.append(trace[j])
            new_trace.append(trace[-1])
            traces_info["traces"][i] = np.stack(new_trace)
        return traces_info

    def reset_traces_info(self):
        if self._traces_info is not None:
            for i in range(len(self._traces_info["traces"])):
                self._traces_info["traces"][i] = []
                self._traces_info["starts"][i] = []
        else:
            raise ValueError("Tracing info is not enabled.")


@heatsplats.register("modules.gpu-geodesic-tracer")
class GPUGeodesicTracer(GeodesicTracer):
    @dataclass
    class Config(GeodesicTracer.Config):
        max_iterations: Optional[int] = None
        avoid_holes: bool = True
        n_debug_traces: int = 10

    cfg: Config

    def configure(self, mesh: Mesh):
        super().configure(mesh)

        if self.debug:
            heatsplats.warn(
                "GPUGeodesicTracer does not support debug mode (trace visualization).",
                "Disabling debug mode.",
            )
            self.debug = False

        if self.cfg.max_iterations is None:
            self.cfg.max_iterations = 2**63 - 1  # like pp3d

        self._digeo_mesh = digeo.Mesh(
            positions=self.mesh.verts.cpu().numpy(),
            triangles=self.mesh.faces.cpu().numpy(),
            adjacencies=None,
            triangle_normals=self.mesh.fnorms.cpu().numpy(),
            v2t=None,
            vertex_normals=self.mesh.vnorms.cpu().numpy(),
            device=self.mesh.verts.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def trace(
        self,
        coords: Float[Tensor, "B 3"],
        face_ids: Int[Tensor, "B"],
        tangent_vector: Float[Tensor, "B 3"],
        *,
        bary_coords: Optional[Float[Tensor, "B 3"]] = None,
        transport_vector: Optional[Float[Tensor, "B 3"]] = None,
        out_coords: Optional[Float[Tensor, "B 3"]] = None,
        out_face_ids: Optional[Float[Tensor, "B"]] = None,
        out_transport_vector: Optional[Float[Tensor, "B 3"]] = None,
    ) -> tuple[Float[Tensor, "B 3"], Int[Tensor, "B"]]:
        needs_parallel_transport = transport_vector is not None

        if bary_coords is None:
            vert_ids = self.mesh.get_face_vertices(face_ids)
            bary_coords = self.mesh.cartesian_to_barycentric(coords, vert_ids)

        start_meshpoints = digeo.MeshPointBatch(face_ids.int(), bary_coords[:, 1:])

        end_meshpoints, geodesic_info = digeo.ops.trace_geodesics(
            self._digeo_mesh,
            start_meshpoints,
            tangent_vector,
            gradient="none",
            use_python=False,
            max_steps=self.cfg.max_iterations,
            save_parallel_transport=needs_parallel_transport,
            save_end_direction=False,
            debug=False,
            print_warnings=True,
            avoid_holes=self.cfg.avoid_holes,
        )

        new_coords = end_meshpoints.interpolate(self._digeo_mesh)
        new_face_ids = end_meshpoints.faces

        if not bool((new_face_ids >= 0).all()):
            bad = torch.nonzero(new_face_ids < 0, as_tuple=False).flatten()[:8].tolist()
            raise AssertionError(
                f"digeo returned invalid face ids (<0). "
                f"num_bad={(new_face_ids < 0).sum().item()} sample_idx={bad}"
            )

        if out_coords is None:
            out_coords = new_coords.to(device=coords.device, dtype=coords.dtype).clone()
        else:
            out_coords.copy_(new_coords)

        if out_face_ids is None:
            out_face_ids = new_face_ids.to(
                device=face_ids.device, dtype=face_ids.dtype
            ).clone()
        else:
            out_face_ids.copy_(new_face_ids)

        if not needs_parallel_transport:
            return out_coords, out_face_ids

        new_transport_vector = geodesic_info.transport(transport_vector)
        if out_transport_vector is None:
            out_transport_vector = new_transport_vector.to(
                device=transport_vector.device, dtype=transport_vector.dtype
            ).clone()
        else:
            out_transport_vector.copy_(new_transport_vector)

        return out_coords, out_face_ids, out_transport_vector
