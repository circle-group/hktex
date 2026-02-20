from dataclasses import dataclass, field
import math

import torch
import torch.linalg as linalg

from tqdm import tqdm

import heatsplats
from heatsplats.utils import BaseObject, interpolate_barycentric_attr_from_trivertidx
from heatsplats.utils.typing import *

from .mesh import Mesh
from .eigen_albo import EigenAlboInterpolation

import heatsplats.knn_heat as knn_heat

__all__ = ["EigenAlboInterpolationKNN"]


@heatsplats.register("modules.eigen-albo-interpolation-knn")
class EigenAlboInterpolationKNN(EigenAlboInterpolation):
    @dataclass
    class Config(EigenAlboInterpolation.Config):
        faiss_config = field(default_factory=knn_heat.FaissGpuIndexConfig)
        gather_config = field(default_factory=knn_heat.SpectralKnnGatherConfig)
        heat_config = field(default_factory=knn_heat.SpectralKnnHeatConfig)

        use_weighting: bool = True
        weighting_config = field(default_factory=knn_heat.KnnPostDiffWeightConfig)

        knn_embedding_dim: int = 64

        grid_abs_sin: bool = True
        grid_scale_map: Literal["log1p", "identity"] = "log1p"
        grid_eps: float = 1e-8
        grid_clamp_unit: bool = True

    cfg: Config

    def configure(self, mesh: Mesh):
        super().configure(mesh)

        self.faiss_index = knn_heat.FaissGpuFlatIndex(self.cfg.faiss_config)
        self.knn_gather = knn_heat.SpectralKnnGather(self.cfg.gather_config)
        self.knn_heat = knn_heat.SpectralKnnHeat(self.cfg.heat_config)

        self.heat_weighting = None
        if self.cfg.use_weighting:
            self.heat_weighting = knn_heat.KnnPostDiffWeight(self.cfg.heat_config)

        self.knn_embedding_dim = self.cfg.knn_embedding_dim
        assert (
            1 <= self.knn_embedding_dim <= self.cfg.k_eig
        ), "KNN embedding dim cannot be larger than the number of eigenvalues/less than 1"

        iso_evals = self._iso_eigen_val[: self.knn_embedding_dim]
        iso_evecs = self._iso_eigen_vec[:, : self.knn_embedding_dim]
        self._iso_embeddings = iso_evecs / iso_evals.unsqueeze(0)

        self.reset()

    def reset(self):
        self.faiss_index.reset()
        self.grid = None
        self.knn_heat.reset()

    def build_kernel_graph(
        self,
        kernel_bary: Float[Tensor, "G 3"],
        kernel_vert_idx: Float[Tensor, "G 3"],
        kernel_angles: Float[Tensor, "G"],
        kernel_scales: Float[Tensor, "G"],
        diffusion_time: Float[Tensor, ""] | Float[Tensor, "G"],
    ):
        # Get the iso kernel embeddings and build the faiss index
        kernel_embeddings = interpolate_barycentric_attr_from_trivertidx(
            kernel_vert_idx, kernel_bary, self._iso_embeddings
        )
        self.faiss_index.build(kernel_embeddings)

        # Create the rotation/anisotrophy interpolation grid indices and weights
        grid_idx, grid_w = self.knn_gather.select_grid(
            self._smp_coords_angle,
            self._smp_coords_scale,
            kernel_angles,
            kernel_scales,
            abs_sin=self.cfg.grid_abs_sin,
            scale_map=self.cfg.grid_scale_map,
            grid_layout=EigenAlboInterpolation.GRID_CONFIGURATION,
            eps=self.cfg.grid_eps,
            clamp_unit=self.cfg.grid_clamp_unit,
        )
        self.grid = (grid_idx, grid_w)
        # Interpolate kernel evals and evecs
        kernel_evals, kernel_evecs = self.knn_gather.gather_src(
            grid_idx,
            grid_w,
            kernel_vert_idx,
            kernel_bary,
            self._eigen_val,
            self._eigen_vec,
        )
        # Build the knn graph
        self.knn_heat.build(kernel_evals, kernel_evecs, diffusion_time)

    def query_points(
        self,
        barycentric_coords: Float[Tensor, "P 3"],
        vert_idx: Int[Tensor, "P 3"],
        knn_k: int = 64,
    ):
        # Calculate knn points
        point_embeddings = interpolate_barycentric_attr_from_trivertidx(
            vert_idx, barycentric_coords, self._iso_embeddings
        )
        _, knn_indices = self.faiss_index.search(point_embeddings, k=knn_k)
        knn_distances = self.faiss_index.knn_distances(point_embeddings, knn_indices)

        # Biharmonic distance weights
        heat_weights = None
        if self.cfg.use_weighting:
            heat_weights = self.heat_weighting.compute(knn_distances)

        # Gather point evals and evecs
        grid_idx, grid_w = self.grid
        evals, evecs, _ = self.knn_gather.gather_queries(
            knn_indices,
            grid_idx,
            grid_w,
            vert_idx,
            barycentric_coords,
            self._eigen_val,
            self._eigen_vec,
        )

        return {
            "evals": evals,  # P K E
            "evecs": evecs,  # P K E
            "weights": heat_weights,  # P K
            "distances": knn_distances,  # P K
            "indices": knn_indices,  # P K
        }

    def diffuse_heat(
        self,
        evals: Float[Tensor, "P K E"],
        evecs: Float[Tensor, "P K E"],
        indices: Int64[Tensor, "P K"],
        weights: Optional[Float[Tensor, "P K"]] = None,
        diffusion_time: Optional[
            Float[Tensor, ""] | Float[Tensor, "P K"] | Float[Tensor, "G"]
        ] = None,
    ):
        self.knn_heat.query(
            indices, evals, evecs, time=diffusion_time, weights_post_diff=weights
        )
