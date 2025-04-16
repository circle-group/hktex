from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import mitsuba as mi

import heatsplats
from heatsplats.data import MeshSamplerDataModule
from heatsplats.rendering.uv_texture_renderer import UVTextureRenderer
from heatsplats.utils.typing import *

from .base import BaseTrainer


@heatsplats.register("trainers.uv-texture")
class UvTextureTrainer(BaseTrainer):
    @dataclass
    class Config(BaseTrainer.Config):
        pass

    cfg: Config

    def configure(
        self,
        datamodule: MeshSamplerDataModule,
        **kwargs,
    ):
        super().configure(datamodule, **kwargs)

    def prepare_batch(self, data: dict) -> dict:
        data = super().prepare_batch(data)
        face_ids = data["face_id"]
        barys = data["bary"]

        pts_tri_vert_idx = self.mesh.get_face_vertices(face_ids)  # [P, 3]

        albo_weights = self.eigalbo_interp.interpolate_anisotropies(
            angles=self.model.angles, scales=self.model.anisotropies
        )
        albo_evals, pts_evecs, pts_mass = self.eigalbo_interp.barycentric_albo_points(
            albo_weights=albo_weights,
            barycentric_coords=barys,
            vert_idx=pts_tri_vert_idx,
        )

        data["evals"] = albo_evals
        data["pts_evecs"] = pts_evecs
        data["pts_mass"] = pts_mass
        data["albo_weights"] = albo_weights

        return data

    def render_gt(self, rotating_frames: int = 10) -> Union[mi.Bitmap, list[mi.Bitmap]]:

        renderer = UVTextureRenderer(self.cfg.renderer)
        mi_mesh = renderer.mesh_to_mitsuba(self.datamodule.mesh)
        if rotating_frames == 1:
            img = renderer.render(mi_mesh, denoise=True)
            out = mi.Bitmap(img).convert(
                pixel_format=mi.Bitmap.PixelFormat.RGB,
                component_format=mi.Struct.Type.UInt8,
                srgb_gamma=True,
            )
        else:
            out = renderer.rotating_video(mi_mesh, rotating_frames)
        return out
