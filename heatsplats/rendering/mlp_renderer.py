import torch
import torch.nn as nn
import drjit as dr
import mitsuba as mi
import numpy as np

from dataclasses import dataclass

import heatsplats.utils as utils
from heatsplats.utils.typing import *
from .base import BaseRenderer
from .util import MitsubaWrapper, vec_to_tens_safe


class MLPTextureNetwork(MitsubaWrapper):
    def __init__(
        self,
        network: nn.Module,
        scene_min: float,
        scene_max: float,
        point_batching: Optional[int] = None,
    ) -> None:
        super().__init__("differentiable_heat_kernels_net")
        self.network = network
        self.scene_min = scene_min
        self.scene_max = scene_max
        self.point_batching = point_batching

    def _eval(self, si, dirs, norms, albedo):
        pts = si.p
        pts = (pts - self.scene_min) / (self.scene_max - self.scene_min)
        pts = 2 * pts + 1
        pts_tensor = vec_to_tens_safe(pts + self.grad_activator)

        torch_out = self._eval_in_torch(pts_tensor, batch_size=self.point_batching)
        output = dr.unravel(mi.Vector3f, torch_out.array)
        return dr.clip(output, 0, 1)

    @dr.wrap(source="drjit", target="torch")
    def _eval_in_torch(self, pts, batch_size=None):
        if batch_size is None:
            return self.network(pts)

        P = pts.shape[0]
        colours = []

        for i in range(0, P, batch_size):
            # Slice the current batch. IT handles also when the smaller batch is smaller
            pts_batch = pts[i : i + batch_size]

            colours_batch = self.network(pts_batch)

            # Store the batch results
            # colours[i : i + batch_size, :] = colours_batch
            colours.append(colours_batch)

        # return colours
        return torch.cat(colours, dim=0)

    def _traverse(self, callback):
        callback.put("network", self.network, mi.ParamFlags.Differentiable)


class MLPTextureRenderer(BaseRenderer):
    """
    MLPTextureRenderer is a specialized renderer for visualizing mlp textures.
    """

    @dataclass
    class Config(BaseRenderer.Config):
        point_batching: Optional[int] = None

    cfg: Config

    def configure(self):
        super().configure()
        self._tile_size = self.cfg.camera_config.tile_size_heatkernels

    def mesh_to_mitsuba(
        self,
        tri_mesh: Trimesh,
        network: nn.Module,
        scene_min: float,
        scene_max: float,
        **kwargs,
    ) -> tuple[mi.Mesh, mi.Texture]:

        network = MLPTextureNetwork(
            network=network,
            scene_min=scene_min,
            scene_max=scene_max,
            point_batching=self.cfg.point_batching,
        )

        hk_texture = mi.load_dict({"type": "torch_texture"})
        hk_texture.network = network

        bsdf_dict = {
            "type": "principled",
            "base_color": hk_texture,
        }

        if self.cfg.mitsuba_mesh_config.twosided:
            bsdf_dict = {"type": "twosided", "material": bsdf_dict}

        bsdf_prop = mi.Properties()
        bsdf_prop["mesh_bsdf"] = mi.load_dict(bsdf_dict)

        mi_mesh = mi.Mesh(
            "mesh",
            vertex_count=tri_mesh.vertices.shape[0],
            face_count=tri_mesh.faces.shape[0],
            props=bsdf_prop,
        )

        # "Traverse" the mesh to get its updateable parameters
        mesh_params = mi.traverse(mi_mesh)
        mesh_params["vertex_positions"] = np.array(tri_mesh.vertices).flatten()
        mesh_params["faces"] = np.array(tri_mesh.faces).flatten()

        mesh_params.update()
        return mi_mesh, hk_texture

    def mesh_notex_to_mitsuba(self, tri_mesh: Trimesh, **kwargs) -> mi.Mesh:
        bsdf_dict = {
            "type": "principled",
        }

        if self.cfg.mitsuba_mesh_config.twosided:
            bsdf_dict = {"type": "twosided", "material": bsdf_dict}

        bsdf_prop = mi.Properties()
        bsdf_prop["mesh_bsdf"] = mi.load_dict(bsdf_dict)

        mi_mesh = mi.Mesh(
            "mesh",
            vertex_count=tri_mesh.vertices.shape[0],
            face_count=tri_mesh.faces.shape[0],
            props=bsdf_prop,
        )

        # "Traverse" the mesh to get its updateable parameters
        mesh_params = mi.traverse(mi_mesh)
        mesh_params["vertex_positions"] = np.array(tri_mesh.vertices).flatten()
        mesh_params["faces"] = np.array(tri_mesh.faces).flatten()

        mesh_params.update()
        return mi_mesh
