import torch
import torch.nn as nn
import drjit as dr
import mitsuba as mi
import numpy as np

from dataclasses import dataclass

import heatsplats.utils as utils
from heatsplats.utils.typing import *
from heatsplats.modules import (
    Mesh,
    HeatKernelTexture,
    EigenAlboInterpolation,
    PointsInfo,
    KernelInfo,
)
from .base import BaseRenderer
from .util import MitsubaWrapper, vec_to_tens_safe


class DifferentiableHeatKernelsNetwork(MitsubaWrapper):
    def __init__(
        self,
        mesh: Mesh,
        model: HeatKernelTexture,
        eigalbo_interp: EigenAlboInterpolation,
        point_batching: int,
    ) -> None:
        super().__init__("differentiable_heat_kernels_net")
        self.mesh: Mesh = mesh
        self.model: HeatKernelTexture = model
        self.eigalbo_interp: EigenAlboInterpolation = eigalbo_interp
        self.point_batching: int = point_batching

    def _eval(self, si, dirs, norms, albedo):
        pts, prim_index = si.p, si.prim_index
        pts_tensor = vec_to_tens_safe(pts + self.grad_activator)

        torch_out = self._eval_in_torch(
            pts_tensor, prim_index, batch_size=self.point_batching
        )
        print("torch_out:", torch_out)
        output = dr.unravel(mi.Vector3f, torch_out.array)
        return output

    @dr.wrap(source="drjit", target="torch")
    def _eval_in_torch(self, pts, face_ids, batch_size=1024):
        print(pts.shape)

        # pts = pts.T
        face_ids = face_ids.to(torch.int)
        P = pts.shape[0]

        # colours: Float[Tensor, "P C"] = torch.zeros(
        #     [P, self.model.out_dim],
        #     device=pts.device,
        #     dtype=pts.dtype,
        # )
        colours = []

        albo_weights = self.eigalbo_interp.interpolate_anisotropies(
            angles=self.model.angles, scales=self.model.anisotropies
        )

        kernel_info: KernelInfo = self.model.prepare_kernels_for_diffusion(
            mesh=self.mesh,
            eigalbo_interp=self.eigalbo_interp,
            albo_weights=albo_weights,
            save_barycentric=False,
        )

        for i in range(0, P, batch_size):
            # Slice the current batch. IT handles also when the smaller batch is smaller

            pts_batch = pts[i : i + batch_size]
            face_ids_batch = face_ids[i : i + batch_size]

            print("here_1", i, i + batch_size)
            # print("\t", pts_batch.shape)

            points_info: PointsInfo = self.model.prepare_points_for_diffusion(
                mesh=self.mesh,
                eigalbo_interp=self.eigalbo_interp,
                albo_weights=albo_weights,
                face_ids=face_ids_batch,
                barys=None,
                pts=pts_batch,
            )

            # print("here_2", i)

            colours_batch, _, _ = self.model.diffuse_heat_kernels(
                eigalbo_interp=self.eigalbo_interp,
                pts_info=points_info,
                kernel_info=kernel_info,
                at_vertices=False,
            )  # [p, D] with p = pts_batch.shape[0]

            # print("here_3", i)

            colours_batch = self.model(colours_batch)  # Postprocess

            # Store the batch results
            # colours[i : i + batch_size, :] = colours_batch
            colours.append(colours_batch)

            # print("here_4", i)
        print("here_for")

        # return colours
        return torch.cat(colours, dim=0)

    def _traverse(self, callback):
        callback.put("model", self.model, mi.ParamFlags.Differentiable)


class DifferentiableHeatKernelsRenderer(BaseRenderer):
    """
    HeatKernelsRenderer is a specialized renderer for visualizing heat kernel textures.
    """

    @dataclass
    class Config(BaseRenderer.Config):
        point_batching: int = 1024

    cfg: Config

    def configure(self):
        super().configure()
        self._tile_size = self.cfg.camera_config.tile_size_heatkernels

    def mesh_to_mitsuba(
        self,
        tri_mesh: Trimesh,
        mesh: Mesh,
        model: HeatKernelTexture,
        eigalbo_interp: EigenAlboInterpolation,
        **kwargs,
    ) -> tuple[mi.Mesh, mi.Texture]:

        network = DifferentiableHeatKernelsNetwork(
            mesh=mesh,
            model=model,
            eigalbo_interp=eigalbo_interp,
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
