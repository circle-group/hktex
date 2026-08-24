from dataclasses import dataclass, field
from abc import abstractmethod
import numpy as np
from termcolor import colored
import trimesh

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import heatsplats

import heatsplats.utils as utils
from heatsplats.utils import BaseModule
from heatsplats.utils.typing import *

from .mesh import Mesh
from .eigen_albo import EigenAlboInterpolation
from .eigen_albo_knn import EigenAlboInterpolationKNN
from .utils import PointsInfoKNN, KernelInfo
from .heat_kernel_texture_knn import HeatKernelTextureKNN

__all__ = ["HeatKernelDensityKNN"]


@heatsplats.register("modules.heat-kernel-density-knn")
class HeatKernelDensityKNN(HeatKernelTextureKNN):
    @dataclass
    class Config(HeatKernelTextureKNN.Config):
        density_mode: str = "coverage"  # "centers" | "coverage"
        center_sigma: float = 0.05
        density_min: float = 0.0
        density_max: float = 1.0
        log_beta: float = 0.0

    cfg: Config

    def configure(self, mesh, **kwargs):
        super().configure(mesh, **kwargs)

    def filtered_kernel_weights(
        self,
        pts_info: PointsInfoKNN,
        eigalbo_interp: EigenAlboInterpolationKNN,
    ) -> tuple[Float[Tensor, "P K"], Int[Tensor, "P K"]]:
        pts_evecs: Float[Tensor, "P K E"] = pts_info["albo_evecs"]
        pts_evals: Float[Tensor, "P K E"] = pts_info["albo_evals"]
        pts_weights: Float[Tensor, "P K"] | None = pts_info["weights"]
        pts_indices: Int[Tensor, "P K"] = pts_info["indices"]

        heat_qk, heat_qk_norm = eigalbo_interp.diffuse_heat(
            pts_evals,
            pts_evecs,
            pts_indices,
            weights=pts_weights,
        )
        diffused_diracs: Float[Tensor, "P K"] = (
            heat_qk_norm if heat_qk_norm is not None else heat_qk
        )

        filtered: Float[Tensor, "P K"] = self.kernel_filter_func(
            diffused_diracs,
            epsilon=self.thresholds[pts_indices],
            sharpness=self.sharpnesses[pts_indices],
        )
        return filtered, pts_indices

    def coverage_density_from_points_info(
        self,
        pts_info: PointsInfoKNN,
        eigalbo_interp: EigenAlboInterpolationKNN,
    ) -> Float[Tensor, "P"]:
        filtered, _ = self.filtered_kernel_weights(pts_info, eigalbo_interp)
        return filtered.sum(dim=1)

    def center_density_from_points_info(
        self,
        pts_info: PointsInfoKNN,
    ) -> Float[Tensor, "P"]:
        dists: Float[Tensor, "P K"] = pts_info["distances"]

        sigma = self.cfg.center_sigma
        weights = torch.exp(-0.5 * (dists / (sigma + 1e-8)) ** 2)
        return weights.sum(dim=1)

    def density_from_points_info(
        self,
        pts_info: PointsInfoKNN,
        eigalbo_interp: EigenAlboInterpolationKNN,
    ) -> Float[Tensor, "P"]:
        if self.cfg.density_mode == "coverage":
            return self.coverage_density_from_points_info(pts_info, eigalbo_interp)

        if self.cfg.density_mode == "centers":
            return self.center_density_from_points_info(pts_info)

        raise ValueError(f"Unknown density mode: {self.cfg.density_mode}")

    def density_to_rgb(
        self,
        density: Float[Tensor, "P"],
    ) -> Float[Tensor, "P 3"]:
        t = (density - self.cfg.density_min) / (
            self.cfg.density_max - self.cfg.density_min + 1e-8
        )
        t = t.clamp(0.0, 1.0)

        if self.cfg.log_beta > 0.0:
            beta = torch.tensor(
                self.cfg.log_beta,
                dtype=t.dtype,
                device=t.device,
            )
            t = torch.log1p(beta * t) / torch.log1p(beta)

        return self.viridis_like(t)

    def viridis_like(
        self,
        t: Float[Tensor, "P"],
    ) -> Float[Tensor, "P 3"]:
        anchors = torch.tensor(
            [
                [0.267, 0.005, 0.329],
                [0.230, 0.322, 0.546],
                [0.128, 0.567, 0.551],
                [0.369, 0.789, 0.383],
                [0.993, 0.906, 0.144],
            ],
            dtype=t.dtype,
            device=t.device,
        )

        x = t.clamp(0.0, 1.0) * (anchors.shape[0] - 1)
        i0 = torch.floor(x).long().clamp(0, anchors.shape[0] - 2)
        i1 = i0 + 1
        a = (x - i0.to(x.dtype)).unsqueeze(-1)

        return anchors[i0] * (1.0 - a) + anchors[i1] * a
