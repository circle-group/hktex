from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import mitsuba as mi

import heatsplats
from heatsplats.data import MeshSamplerDataModule
from heatsplats.modules import PointsInfo
from heatsplats.rendering.uv_texture_renderer import UVTextureRenderer
from heatsplats.utils import load_mesh
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

        with torch.profiler.record_function("interpolate_anisotropies"):
            albo_weights = self.eigalbo_interp.interpolate_anisotropies(
                angles=self.model.angles, scales=self.model.anisotropies
            )

        with torch.profiler.record_function("prepare_points_for_diffusion"):
            points_info: PointsInfo = self.model.prepare_points_for_diffusion(
                mesh=self.mesh,
                eigalbo_interp=self.eigalbo_interp,
                albo_weights=albo_weights,
                face_ids=face_ids,
                barys=barys,
                pts=None,
            )

        data["evals"] = points_info["albo_evals"]
        data["pts_iso_evecs"] = points_info["iso_evecs"]
        data["pts_evecs"] = points_info["albo_evecs"]
        data["pts_mass"] = points_info["mass"]
        data["albo_weights"] = albo_weights
        data["points_info"] = points_info

        return data

    def render_gt(self, rotating_frames: int = 10) -> Union[mi.Bitmap, list[mi.Bitmap]]:

        renderer = UVTextureRenderer(self.cfg.renderer)
        mesh = self.datamodule.mesh

        if self.datamodule.cfg.merge_tex:
            # Reload the mesh without merging textures to get proper UVs
            mesh = load_mesh(
                self.datamodule.cfg.mesh_path,
                show=False,
                merge_tex=False,
                bake_vert_colors=False,
            )

        mi_mesh = renderer.mesh_to_mitsuba(mesh)
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
