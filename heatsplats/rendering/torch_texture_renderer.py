import torch
import torch.nn as nn
import drjit as dr
import mitsuba as mi
import numpy as np

from dataclasses import dataclass

import heatsplats
from heatsplats.modules.base import TextureModel
import heatsplats.utils as utils
from heatsplats.utils.typing import *
from .base import BaseRenderer
from .util import MitsubaWrapper, vec_to_tens_safe


@heatsplats.register("texture.torch-network")
class TorchTextureNetwork(MitsubaWrapper):
    def __init__(
        self,
        network: TextureModel,
        point_batching: Optional[int] = None,
    ) -> None:
        super().__init__("torch_texture_net")
        self.network = network
        self.point_batching = point_batching

    def _eval(self, si, dirs, norms, albedo):
        pts = si.p
        pts_tensor = vec_to_tens_safe(pts + self.grad_activator)
        kwargs = dict()
        if self.network.requires_face_ids():
            kwargs["face_ids"] = si.prim_index

        torch_out = self._eval_in_torch(
            pts_tensor, batch_size=self.point_batching, **kwargs
        )
        output = dr.unravel(mi.Vector3f, torch_out.array)
        return dr.clip(output, 0, 1)

    @dr.wrap(source="drjit", target="torch")
    def _eval_in_torch(self, pts, batch_size=None, **kwargs):
        if batch_size is None:
            return self.network(pts, **kwargs)

        P = pts.shape[0]
        colours = []

        for i in range(0, P, batch_size):
            # Slice the current batch. IT handles also when the smaller batch is smaller
            pts_batch = pts[i : i + batch_size]

            colours_batch = self.network(pts_batch, **kwargs)

            # Store the batch results
            # colours[i : i + batch_size, :] = colours_batch
            colours.append(colours_batch)

        # return colours
        return torch.cat(colours, dim=0)

    def _traverse(self, callback):
        callback.put("network", self.network, mi.ParamFlags.Differentiable)


@heatsplats.register("renderer.torch-texture")
class TorchTextureRenderer(BaseRenderer):
    """
    TorchTextureRenderer is a specialized renderer for visualizing torch network textures.
    """

    @dataclass
    class Config(BaseRenderer.Config):
        network_type: str = "texture.torch-network"

        point_batching: Optional[int] = None

    cfg: Config

    def configure(self):
        super().configure()
        self._tile_size = self.cfg.camera_config.tile_size_heatkernels

    def mesh_to_mitsuba(
        self,
        tri_mesh: Trimesh,
        network: TextureModel,
        **kwargs,
    ) -> tuple[mi.Mesh, mi.Texture]:

        network: TorchTextureNetwork = heatsplats.find(self.cfg.network_type)(
            network=network,
            point_batching=self.cfg.point_batching,
        )

        hk_texture = mi.load_dict({"type": "torch_texture"})
        hk_texture.network = network

        mi_mesh = self.mesh_notex_to_mitsuba(tri_mesh, base_color=hk_texture)

        return mi_mesh, hk_texture

    def mesh_notex_to_mitsuba(self, tri_mesh: Trimesh, **kwargs) -> mi.Mesh:
        bsdf_dict = {"type": "principled", **kwargs}

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
