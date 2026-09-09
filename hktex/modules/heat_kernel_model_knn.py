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

import hktex

from hktex.modules.heat_kernel_texture_knn import HeatKernelTextureKNN
from hktex.modules.eigen_albo_knn import EigenAlboInterpolationKNN
from hktex.modules.heat_kernel_model import HeatKernelModel
from hktex.modules.utils import PointsInfo
from hktex.utils.typing import *

__all__ = ["HeatKernelModelKNN"]


@hktex.register("modules.heat-kernel-model-knn")
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
        self._inference_mode = False

        assert isinstance(
            self.eigalbo_interp, EigenAlboInterpolationKNN
        ), "Incorrect EigenAlbo Interpolator Type"

    def inference_mode(self, new_mode: bool) -> bool:
        old_mode = self._inference_mode
        self._inference_mode = new_mode
        return old_mode

    def _make_model(self):
        return HeatKernelTextureKNN(self.cfg.model, self.mesh)

    def prepare_kernels(self):
        return self.model.prepare_kernels(
            self.mesh, self.eigalbo_interp, save_barycentric=True
        )

    def reset(self):
        return self.model.reset(self.eigalbo_interp)

    def forward(
        self, pts: Float[Tensor, "P in_dim"], **kwargs
    ) -> Float[Tensor, "P out_dim"]:
        if not self._inference_mode:
            self.prepare_kernels()

        face_ids: Tensor = kwargs["face_ids"].to(torch.int)
        P = pts.shape[0]
        assert P == face_ids.shape[0]
        batch_size = self.cfg.point_batching

        if batch_size is None:
            out = self._forward_batch(pts, face_ids)
        else:
            colours = []
            for i in range(0, P, batch_size):
                pts_batch = pts[i : i + batch_size]
                face_ids_batch = face_ids[i : i + batch_size]
                colours_batch = self._forward_batch(pts_batch, face_ids_batch)
                colours.append(colours_batch)
            out = torch.cat(colours, dim=0)

        return out

    def _forward_batch(self, pts_batch, face_ids_batch):
        points_info: PointsInfo = self.model.prepare_points(
            mesh=self.mesh,
            eigalbo_interp=self.eigalbo_interp,
            face_ids=face_ids_batch,
            barys=None,
            pts=pts_batch,
        )
        colours_batch, _, _, _ = self.model.diffuse_heat_kernels(
            eigalbo_interp=self.eigalbo_interp, pts_info=points_info
        )  # [p, D] with p = pts_batch.shape[0]
        colours_batch = self.model(colours_batch)  # Postprocess
        return colours_batch
