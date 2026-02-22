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

from heatsplats.modules.heat_kernel_texture_knn import HeatKernelTextureKNN
from heatsplats.modules.eigen_albo_knn import EigenAlboInterpolationKNN
from heatsplats.modules.heat_kernel_model import HeatKernelModel
from heatsplats.modules.utils import PointsInfo
from heatsplats.utils.typing import *


__all__ = ["HeatKernelModelKNN"]


@heatsplats.register("modules.heat-kernel-model-knn")
class HeatKernelModelKNN(HeatKernelModel):
    @dataclass
    class Config(HeatKernelModel.Config):
        pass

    cfg: Config

    eigalbo_interp: EigenAlboInterpolationKNN

    def configure(
        self,
        **kwargs,
    ):
        super().configure(**kwargs)

        # TODO: make this dynamic in base class
        self.model = HeatKernelTextureKNN(self.cfg.model, self.mesh)

        assert isinstance(
            self.eigalbo_interp, EigenAlboInterpolationKNN
        ), "Incorrect EigenAlbo Interpolator Type"

    def prepare_kernels(self):
        return self.model.prepare_kernels(
            self.mesh, self.eigalbo_interp, save_barycentric=True
        )

    def forward(
        self, pts: Float[Tensor, "P in_dim"], **kwargs
    ) -> Float[Tensor, "P out_dim"]:
        face_ids: Tensor = kwargs["face_ids"].to(torch.int)
        P = pts.shape[0]
        batch_size = self.cfg.point_batching

        if batch_size is None:
            return self._forward_batch(pts, face_ids)

        colours = []
        for i in range(0, P, batch_size):
            pts_batch = pts[i : i + batch_size]
            face_ids_batch = face_ids[i : i + batch_size]
            colours_batch = self._forward_batch(pts_batch, face_ids_batch)
            colours.append(colours_batch)
        colours = torch.cat(colours, dim=0)
        return colours

    def _forward_batch(self, pts_batch, face_ids_batch):
        points_info: PointsInfo = self.model.prepare_points(
            mesh=self.mesh,
            eigalbo_interp=self.eigalbo_interp,
            face_ids=face_ids_batch,
            barys=None,
            pts=pts_batch,
        )
        colours_batch, _, _ = self.model.diffuse_heat_kernels(
            eigalbo_interp=self.eigalbo_interp, pts_info=points_info
        )  # [p, D] with p = pts_batch.shape[0]
        colours_batch = self.model(colours_batch)  # Postprocess
        return colours_batch
