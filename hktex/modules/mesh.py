from dataclasses import dataclass, field
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F

import hktex
from hktex.utils import cart_to_bary_coords, bary_to_cart_coords, compute_tot_area
from hktex.utils.typing import *

__all__ = ["Mesh"]


class Mesh(nn.Module):
    verts: Float[Tensor, "V 3"]
    faces: Int[Tensor, "F 3"]
    fnorms: Float[Tensor, "F 3"]
    tot_area: Float[Tensor, "1"]

    N_verts: int
    N_faces: int

    def __init__(
        self,
        vertices: Float[Any, "V 3"],
        faces: Int[Any, "F 3"],
        fnorms: Float[Any, "F 3"],
        vnorms: Optional[Float[Any, "V 3"]],
        uv: Optional[Float[Any, "V 2"]] = None,
        device=None,
    ):
        super().__init__()

        self.verts = nn.Buffer(torch.tensor(vertices, dtype=torch.float, device=device))
        # TODO: Check 32 bit vs 64 bit below
        # self.faces = nn.Buffer(torch.tensor(faces, dtype=torch.int64, device=device))
        self.faces = nn.Buffer(torch.tensor(faces, dtype=torch.int32, device=device))
        self.fnorms = nn.Buffer(torch.tensor(fnorms, dtype=torch.float, device=device))
        self.vnorms = nn.Buffer(torch.tensor(vnorms, dtype=torch.float, device=device))
        self.tot_area = compute_tot_area(self.verts, self.faces.T)
        self.uv = uv

        self.N_verts = self.verts.shape[0]
        self.N_faces = self.faces.shape[0]
        assert self.fnorms.shape[0] == self.N_faces

    def get_face_vertices(self, face_ids: Int[Tensor, "B"]) -> Float[Tensor, "B 3"]:
        return self.faces[face_ids]

    def get_vertex_locations(
        self, vert_ids: Int[Tensor, "B 3"]
    ) -> Float[Tensor, "B 3 3"]:
        assert vert_ids.shape[1] == 3
        return self.verts[vert_ids.flatten()].view(-1, 3, 3)

    def cartesian_to_barycentric(
        self, coords: Float[Tensor, "B 3"], vert_ids: Int[Tensor, "B 3"]
    ) -> Float[Tensor, "B 3"]:
        assert coords.shape[0] == vert_ids.shape[0] and coords.shape[1] == 3
        face_vertx = self.get_vertex_locations(vert_ids)
        return cart_to_bary_coords(coords, face_vertx)

    def barycentric_to_cartesian(
        self, bary_coords: Float[Tensor, "B 3"], vert_ids: Int[Tensor, "B 3"]
    ) -> Float[Tensor, "B 3"]:
        assert bary_coords.shape[0] == vert_ids.shape[0] and bary_coords.shape[1] == 3
        face_vertx = self.get_vertex_locations(vert_ids)
        return bary_to_cart_coords(bary_coords, face_vertx)

    @staticmethod
    def from_trimesh(trimesh: trimesh.Trimesh, **kwargs):
        try:
            uv = torch.tensor(trimesh.visual.uv)
        except AttributeError:
            uv = None

        return Mesh(
            vertices=trimesh.vertices,
            faces=trimesh.faces,
            fnorms=trimesh.face_normals,
            vnorms=trimesh.vertex_normals,
            uv=uv,
            **kwargs
        )
