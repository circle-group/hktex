import torch
import drjit
import mitsuba as mi
import numpy as np

from dataclasses import dataclass

import heatsplats.utils as utils
from heatsplats.utils.typing import *
from heatsplats.modules import Mesh, Model, EigenAlboInterpolation
from .base import BaseRenderer


class HeatKernelsTexture(mi.Texture):
    def __init__(self, props: mi.Properties) -> None:
        mi.Texture.__init__(self, props)
        self.mesh: Mesh = None
        self.model: Model = None
        self.eigalbo_interp: EigenAlboInterpolation = None

    def eval(self, si, active=True, dirs=None, norms=None, albedo=None):
        mi_out = self._eval_in_torch(si.p, si.prim_index)
        return mi.Vector3f(mi_out)

    @drjit.wrap(source="drjit", target="torch")
    @torch.no_grad()
    def _eval_in_torch(self, pts, face_ids, batch_size=1024):

        pts = pts.T
        face_ids = face_ids.to(torch.int)
        P = pts.shape[0]
        B = self.model.N_sources

        colours: Float[Tensor, "P C"] = torch.zeros(
            [P, self.model.out_dim],
            device=pts.device,
        )

        for i in range(0, P, batch_size):
            # Slice the current batch. IT handles also when the smaller batch is smaller

            pts_batch = pts[i : i + batch_size]
            face_ids_batch = face_ids[i : i + batch_size]

            pts_tri_vert_idx = self.mesh.get_face_vertices(face_ids_batch)
            pts_barys = self.mesh.cartesian_to_barycentric(pts_batch, pts_tri_vert_idx)

            albo_weights = self.eigalbo_interp.interpolate_anisotropies(
                angles=self.model.angles, scales=self.model.anisotropies
            )
            evals, pts_evecs, pts_mass = self.eigalbo_interp.barycentric_albo_points(
                albo_weights=albo_weights,
                barycentric_coords=pts_barys,
                vert_idx=pts_tri_vert_idx,
            )

            kernel_vert_idx = self.mesh.get_face_vertices(self.model.kernel_face_ids)
            barycentric_coords = self.mesh.cartesian_to_barycentric(
                self.model.kernel_locations, kernel_vert_idx
            )
            self.model.save_barycentric_locations(barycentric_coords)

            kernel_evecs, kernel_mass = self.eigalbo_interp.barycentric_albo_gaussians(
                albo_weights=albo_weights,
                barycentric_coords=barycentric_coords,
                vert_idx=kernel_vert_idx,
            )

            colours_batch = torch.zeros(
                [B, pts_batch.shape[0], self.model.kernel_dim],
                device=pts.device,
            )

            colours_batch: Float[Tensor, "B p+1 L"] = torch.cat(
                (colours_batch, self.model.kernel_colours.unsqueeze(1)), dim=1
            )
            pts_evecs: Float[Tensor, "B p+1 K"] = torch.cat(
                ((pts_evecs, kernel_evecs.unsqueeze(1))), dim=1
            )
            pts_mass: Float[Tensor, "B p+1"] = torch.cat(
                (pts_mass.expand(B, -1), kernel_mass.unsqueeze(-1)), dim=1
            )

            colours_batch = utils.heat_diffusion_reduce(
                colours_batch,
                pts_mass,
                evals,
                pts_evecs,
                self.model.diff_times,
            )
            colours_batch = colours_batch[: pts_batch.shape[0]]
            colours_batch = self.model(colours_batch)

            # Store the batch results
            colours[i : i + batch_size, :] = colours_batch

        return colours

    def eval_1(self, si, active=True):
        return mi.Float(self.eval(si)[0])

    def to_string(self):
        return "HeatKernelTexture"


mi.register_texture("heat_kernels_texture", lambda p: HeatKernelsTexture(p))


class HeatKernelsRenderer(BaseRenderer):
    """
    HeatKernelsRenderer is a specialized renderer for visualizing heat kernel textures.
    """

    @dataclass
    class Config(BaseRenderer.Config):
        pass

    cfg: Config

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
