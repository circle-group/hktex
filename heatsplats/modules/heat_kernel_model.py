from dataclasses import dataclass, field
from abc import abstractmethod
import numpy as np
from termcolor import colored
import trimesh
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import heatsplats

import heatsplats.utils as utils
from heatsplats.modules.base import TextureModel
from heatsplats.modules.heat_kernel_texture import HeatKernelTexture
from heatsplats.modules.eigen_albo import EigenAlboInterpolation
from heatsplats.modules.tracer import GeodesicTracer
from heatsplats.modules.utils import KernelInfo, PointsInfo
from heatsplats.utils.typing import *


__all__ = ["HeatKernelModel"]


@heatsplats.register("modules.heat-kernel-model")
class HeatKernelModel(TextureModel):
    @dataclass
    class Config(TextureModel.Config):
        eigen_albo_type: str = "modules.eigen-albo-interpolation"
        eigen_albo: dict = field(default_factory=dict)

        model: dict = field(default_factory=dict)

        tracer_type: str = ""
        tracer: dict = field(default_factory=dict)

        point_batching: Optional[int] = None

    cfg: Config

    def configure(
        self,
        **kwargs,
    ):
        super().configure()

        self.mesh = kwargs["mesh"]
        assert self.mesh is not None

        self.model = HeatKernelTexture(self.cfg.model, self.mesh)

        EigenAlboClass = heatsplats.find(self.cfg.eigen_albo_type)
        self.eigalbo_interp: EigenAlboInterpolation = EigenAlboClass(
            self.cfg.eigen_albo, self.mesh
        )

        if (
            "n_debug_traces" in self.cfg.tracer
            and self.cfg.tracer["n_debug_traces"] > self.model.N_sources
        ):
            self.cfg.tracer["n_debug_traces"] = self.model.N_sources
            heatsplats.warn(
                "Number of debug traces should not exceed number of sources. Displaying all sources instead."
            )

        self.tracer: GeodesicTracer = heatsplats.find(self.cfg.tracer_type)(
            self.cfg.tracer, self.mesh
        )

    def requires_scene_bounds(self):
        return False

    def requires_face_ids(self):
        return True

    def forward(
        self, pts: Float[Tensor, "P in_dim"], **kwargs
    ) -> Float[Tensor, "P out_dim"]:
        face_ids: Tensor = kwargs["face_ids"].to(torch.int)
        P = pts.shape[0]
        batch_size = self.cfg.point_batching

        albo_weights = self.eigalbo_interp.interpolate_anisotropies(
            angles=self.model.angles, scales=self.model.anisotropies
        )

        kernel_info: KernelInfo = self.model.prepare_kernels_for_diffusion(
            mesh=self.mesh,
            eigalbo_interp=self.eigalbo_interp,
            albo_weights=albo_weights,
            save_barycentric=False,
        )

        if batch_size is None:
            return self._forward_batch(pts, face_ids, albo_weights, kernel_info)

        colours = []
        for i in range(0, P, batch_size):
            pts_batch = pts[i : i + batch_size]
            face_ids_batch = face_ids[i : i + batch_size]
            colours_batch = self._forward_batch(
                pts_batch, face_ids_batch, albo_weights, kernel_info
            )
            colours.append(colours_batch)
        colours = torch.cat(colours, dim=0)
        return colours

    def _forward_batch(self, pts_batch, face_ids_batch, albo_weights, kernel_info):
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
        return colours_batch

    def save_torch(self, filename):
        self.model.save_torch(filename)

    def save_numpy_npz(self, filename):
        self.model.save_numpy_npz(filename)

    def load_torch(self, filename):
        self.model.load_state_dict(filename)
