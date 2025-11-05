import torch
import drjit as dr
import mitsuba as mi
import numpy as np

from dataclasses import dataclass

import heatsplats.utils as utils
from heatsplats.utils.typing import *
from heatsplats.modules import (
    Mesh,
    Model,
    EigenAlboInterpolation,
    PointsInfo,
    KernelInfo,
)
from .base import BaseRenderer


class HeatKernelsTexture(mi.Texture):
    def __init__(self, props: mi.Properties) -> None:
        mi.Texture.__init__(self, props)
        self.mesh: Mesh = None
        self.model: Model = None
        self.eigalbo_interp: EigenAlboInterpolation = None
        self.point_batching: int = None

    def eval(self, si, active=True, dirs=None, norms=None, albedo=None):
        with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, False), dr.scoped_set_flag(
            dr.JitFlag.SymbolicCalls, False
        ):
            dr.eval(si.p, si.prim_index)
            mi_out: dr.scalar.TensorXf = self._eval_in_torch(
                si.p, si.prim_index, batch_size=self.point_batching
            )
            return dr.unravel(mi.Vector3f, mi_out.array)

    @dr.wrap(source="drjit", target="torch")
    @torch.no_grad()
    def _eval_in_torch(self, pts, face_ids, batch_size=1024):
        # print(pts.shape)

        pts = pts.T
        face_ids = face_ids.to(torch.int)
        P = pts.shape[0]

        colours: Float[Tensor, "P C"] = torch.zeros(
            [P, self.model.out_dim],
            device=pts.device,
            dtype=pts.dtype,
        )

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

            points_info: PointsInfo = self.model.prepare_points_for_diffusion(
                mesh=self.mesh,
                eigalbo_interp=self.eigalbo_interp,
                albo_weights=albo_weights,
                face_ids=face_ids_batch,
                barys=None,
                pts=pts_batch,
            )

            colours_batch, _ = self.model.diffuse_heat_kernels(
                eigalbo_interp=self.eigalbo_interp,
                pts_info=points_info,
                kernel_info=kernel_info,
                at_vertices=False,
            )  # [p, D] with p = pts_batch.shape[0]

            colours_batch = self.model(colours_batch)  # Postprocess

            # Store the batch results
            colours[i : i + batch_size, :] = colours_batch

        return colours

    def to_string(self):
        return "HeatKernelTexture"


mi.register_texture("heat_kernels_texture", lambda p: HeatKernelsTexture(p))


class HeatKernelsRenderer(BaseRenderer):
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
        model: Model,
        eigalbo_interp: EigenAlboInterpolation,
        **kwargs
    ) -> mi.Mesh:

        hk_texture = mi.load_dict({"type": "heat_kernels_texture"})
        hk_texture.model = model
        hk_texture.mesh = mesh
        hk_texture.eigalbo_interp = eigalbo_interp
        hk_texture.point_batching = self.cfg.point_batching

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
        return mi_mesh
