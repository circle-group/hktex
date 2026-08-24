import torch
import drjit as dr
import mitsuba as mi
import numpy as np

from dataclasses import dataclass

import heatsplats.utils as utils
from heatsplats.utils.typing import *
from heatsplats.modules import (
    Mesh,
    HeatKernelDensityKNN,
    EigenAlboInterpolationKNN,
    PointsInfo,
    KernelInfo,
)
from .base import BaseRenderer


class HeatKernelsDensityTextureKNN(mi.Texture):
    def __init__(self, props: mi.Properties) -> None:
        mi.Texture.__init__(self, props)
        self.mesh: Mesh = None
        self.model: HeatKernelDensityKNN = None
        self.eigalbo_interp: EigenAlboInterpolationKNN = None
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
            [P, 3],
            device=pts.device,
            dtype=pts.dtype,
        )

        for i in range(0, P, batch_size):
            # Slice the current batch. IT handles also when the smaller batch is smaller

            pts_batch = pts[i : i + batch_size]
            face_ids_batch = face_ids[i : i + batch_size]

            points_info: PointsInfo = self.model.prepare_points(
                mesh=self.mesh,
                eigalbo_interp=self.eigalbo_interp,
                face_ids=face_ids_batch,
                barys=None,
                pts=pts_batch,
            )

            density_batch: Float[Tensor, "p"] = self.model.density_from_points_info(
                points_info,
                eigalbo_interp=self.eigalbo_interp,
            )

            colours_batch: Float[Tensor, "p 3"] = self.model.density_to_rgb(
                density_batch
            )

            # Store the batch results
            colours[i : i + batch_size, :] = colours_batch

        return colours

    def to_string(self):
        return "HeatKernelDensityTexture"


mi.register_texture(
    "heat_kernels_density_texture_knn", lambda p: HeatKernelsDensityTextureKNN(p)
)


class HeatKernelsDensityRendererKNN(BaseRenderer):
    """
    HeatKernelsRenderer is a specialized renderer for visualizing heat kernel textures.
    """

    @dataclass
    class Config(BaseRenderer.Config):
        point_batching: int = 1024
        use_diffuse_bsdf: bool = True

    cfg: Config

    def configure(self):
        super().configure()
        self._tile_size = self.cfg.camera_config.tile_size_heatkernels

    def mesh_to_mitsuba(
        self,
        tri_mesh: Trimesh,
        mesh: Mesh,
        model: HeatKernelDensityKNN,
        eigalbo_interp: EigenAlboInterpolationKNN,
        **kwargs
    ) -> mi.Mesh:

        density_texture = mi.load_dict({"type": "heat_kernels_density_texture_knn"})
        density_texture.model = model
        density_texture.mesh = mesh
        density_texture.eigalbo_interp = eigalbo_interp
        density_texture.point_batching = self.cfg.point_batching

        if self.cfg.use_diffuse_bsdf:
            bsdf_dict = {
                "type": "diffuse",
                "reflectance": density_texture,
            }
        else:
            bsdf_dict = {
                "type": "principled",
                "base_color": density_texture,
                "specular": 0.0,
                "roughness": 0.6,
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
