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
from .heat_kernel_texture import HeatKernelTexture

__all__ = ["HeatKernelTextureKNN"]


class HeatKernelTextureKNN(HeatKernelTexture):
    @dataclass
    class Config(HeatKernelTexture.Config):
        knn_outer_k: int = 100
        knn_inner_k: Optional[int] = 10

    cfg: Config

    def configure(
        self,
        mesh: Mesh,
        **kwargs,
    ):
        super().configure(mesh, **kwargs)

    @property
    def diff_times(self) -> Float[Tensor, "G"]:
        return torch.scalar_tensor(self.cfg.diff_time, device=self.device)

    def reset(
        self,
        eigalbo_knn: EigenAlboInterpolationKNN,
    ):
        eigalbo_knn.reset()

    def prepare_kernels(
        self,
        mesh: Mesh,
        eigalbo_knn: EigenAlboInterpolationKNN,
        save_barycentric: bool = True,
    ) -> KernelInfo:
        kernel_vert_idx = mesh.get_face_vertices(self.kernel_face_ids)
        kernel_barycentric_coords = mesh.cartesian_to_barycentric(
            self.kernel_locations, kernel_vert_idx
        )

        if save_barycentric:
            self.save_barycentric_locations(kernel_barycentric_coords)

        with torch.profiler.record_function("build_kernel_graph"):
            eigalbo_knn.build_kernel_graph(
                kernel_bary=kernel_barycentric_coords,
                kernel_vert_idx=kernel_vert_idx,
                kernel_angles=self.angles,
                kernel_scales=self.anisotropies,
                diffusion_time=self.diff_times,
            )
        return KernelInfo(
            vert_idx=kernel_vert_idx,
            barycentric_coords=kernel_barycentric_coords,
            albo_evecs=None,
            mass=None,
        )

    def prepare_points(
        self,
        mesh: Mesh,
        eigalbo_interp: EigenAlboInterpolationKNN,
        face_ids: Float[Tensor, "P"],
        barys: Float[Tensor, "P 3"] | None = None,
        pts: Float[Tensor, "P 3"] | None = None,
    ) -> PointsInfoKNN:
        pts_tri_vert_idx = mesh.get_face_vertices(face_ids)  # [P, 3]

        if barys is None and pts is not None:
            barys = mesh.cartesian_to_barycentric(pts, pts_tri_vert_idx)
        elif barys is not None and pts is None:
            pass
        else:
            raise ValueError(
                "Either barys or pts must be provided to prepare points for diffusion"
            )

        k_search = min(self.cfg.knn_outer_k, self.N_sources)
        with torch.profiler.record_function("query_points"):
            query_points = eigalbo_interp.query_points(
                barys, pts_tri_vert_idx, k_search
            )

        return PointsInfoKNN(
            albo_evals=query_points["evals"],
            albo_evecs=query_points["evecs"],
            iso_evecs=None,
            mass=None,
            weights=query_points["weights"],
            distances=query_points["distances"],
            indices=query_points["indices"],
        )

    def diffuse_heat_kernels(
        self,
        eigalbo_interp: EigenAlboInterpolationKNN,
        pts_info: PointsInfoKNN,
    ) -> Tuple[
        Float[Tensor, "P D"], None, Float[Tensor, "k P 1"], Float[Tensor, "k P 1"]
    ]:
        pts_evecs: Float[Tensor, "P K E"] = pts_info["albo_evecs"]
        pts_evals: Float[Tensor, "P K E"] = pts_info["albo_evals"]
        pts_weights: Float[Tensor, "P K"] | None = pts_info["weights"]
        pts_distances: Float[Tensor, "P K"] = pts_info["distances"]
        pts_indices: Float[Tensor, "P K"] = pts_info["indices"]

        P, K, _ = pts_evecs.shape
        knn_inner_k = self.cfg.knn_inner_k
        if knn_inner_k is None:
            knn_inner_k = K

        with torch.profiler.record_function("diffuse_heat"):
            heat_qk, heat_qk_norm = eigalbo_interp.diffuse_heat(
                pts_evals, pts_evecs, pts_indices, weights=pts_weights
            )
        diffused_diracs: Float[Tensor, "P K"] = (
            heat_qk_norm if heat_qk_norm is not None else heat_qk
        )

        if self.cfg.power_diffused_diracs != 1:
            diffused_diracs = diffused_diracs**self.cfg.power_diffused_diracs

        with torch.profiler.record_function("kernel_filter"):
            filtered: Float[Tensor, "P K"] = self.kernel_filter_func(
                diffused_diracs,
                epsilon=self.thresholds[pts_indices],
                sharpness=self.sharpnesses[pts_indices],
            )

        with torch.profiler.record_function("inner_knn_reduce"):
            kernel_colours_knn = self.kernel_colours[pts_indices]  # [P, K, D]
            colours: Float[Tensor, "P K D"] = (
                filtered.unsqueeze(-1) * kernel_colours_knn
            )

            contribs: Float[Tensor, "P k"]
            k_use = min(knn_inner_k, K)
            contribs, topk_local = filtered.topk(k=k_use, largest=True, dim=1)

            idx_exp = topk_local.unsqueeze(-1).expand(
                -1, -1, colours.size(-1)
            )  # [P, k, D]
            contrib_colours: Float[Tensor, "P k D"] = torch.gather(
                colours, dim=1, index=idx_exp
            )
            # colours: Float[Tensor, "P D"] = contrib_colours.sum(dim=1) / (
            #     contribs.sum(dim=1, keepdim=True) + 1e-8
            # )
            colours: Float[Tensor, "P D"] = contrib_colours.sum(dim=1) / torch.clamp(
                contribs.sum(dim=1, keepdim=True), min=1.0
            )

            topk_global = torch.gather(pts_indices, dim=1, index=topk_local)  # [P,k]
            kernel_contributions = None
            topk_kernel_idxs = topk_global.transpose(0, 1).unsqueeze(-1)  # [k,P,1]
            topk_kernel_contribs = contribs.transpose(0, 1).unsqueeze(-1)  # [k,P,1]

        colours = (self._mean_colour + colours).clamp(min=0.0, max=1.0)
        return colours, kernel_contributions, topk_kernel_idxs, topk_kernel_contribs
