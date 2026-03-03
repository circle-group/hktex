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
        faiss: knn_heat.FaissGpuIndexConfig = field(
            default_factory=knn_heat.FaissGpuIndexConfig
        )
        gather: knn_heat.SpectralKnnGatherConfig = field(
            default_factory=knn_heat.SpectralKnnGatherConfig
        )
        heat: knn_heat.SpectralKnnHeatConfig = field(
            default_factory=knn_heat.SpectralKnnHeatConfig
        )

        use_weighting: bool = True
        weighting: knn_heat.KnnPostDiffWeightConfig = field(
            default_factory=knn_heat.KnnPostDiffWeightConfig
        )
        parse_weighting_from_str: bool = False

        knn_embedding_dim: int = 64
        use_euclidian_distance: bool = False

        grid_abs_sin: bool = True
        grid_scale_map: str = "log1p"  # Literal["log1p", "identity"]
        grid_eps: float = 1e-8
        grid_clamp_unit: bool = True

    cfg: Config

    def configure(self, mesh: Mesh):
        super().configure(mesh)

        self.faiss_index = knn_heat.FaissGpuFlatIndex(self.cfg.faiss)
        self.knn_gather = knn_heat.SpectralKnnGather(self.cfg.gather)
        self.knn_heat = knn_heat.SpectralKnnHeat(self.cfg.heat)

        self.heat_weighting = None
        self.use_weighting = self.cfg.use_weighting
        if self.cfg.use_weighting:
            weighting = self.cfg.weighting
            if self.cfg.parse_weighting_from_str:
                weighting = self._parse_weighting_from_str()
            if weighting is None:
                self.use_weighting = False
            else:
                self.heat_weighting = knn_heat.KnnPostDiffWeight(weighting)

        self.knn_embedding_dim = self.cfg.knn_embedding_dim
        max_dim = self.cfg.k_eig - 1

        assert (
            1 <= self.knn_embedding_dim <= max_dim
        ), "KNN embedding dim cannot be larger than the (number of eigenvalues-1)/less than 1"

        if self.cfg.use_euclidian_distance:
            self._iso_embeddings = self._mesh.verts
            self.knn_embedding_dim = 3
        else:
            iso_evals = self._iso_eigen_val[1 : self.knn_embedding_dim + 1]
            iso_evecs = self._iso_eigen_vec[:, 1 : self.knn_embedding_dim + 1]
            self._iso_embeddings = iso_evecs / iso_evals.clamp_min(1e-8).unsqueeze(0)

        s = self._smp_coords[1:]  # drop iso
        A = len(self.cfg.precompute_anisotropies)  # get layout from config
        self._smp_coords_angle = s[::A, 0].contiguous()  # rot axis, length R
        self._smp_coords_scale = s[:A, 1].contiguous()  # scale axis, length A

        self._eigen_val_knn = self._eigen_val[1:].contiguous()
        self._eigen_vec_knn = self._eigen_vec[1:].contiguous()

        self.reset()

    def _parse_weighting_from_str(self):
        std = 1
        if self.cfg.distance_weighting == "inverse":
            kernel = "inverse"
            normalize = True
        elif "gaussian" in self.cfg.distance_weighting:
            kernel = "gaussian"
            normalize = False
            std = float(self.cfg.distance_weighting.split("_")[-1])
            assert std > 0, "Standard deviation must be positive"
        elif self.cfg.distance_weighting == "none":
            return None
        return knn_heat.KnnPostDiffWeightConfig(
            kernel=kernel,
            normalize=normalize,
            gaussian_std=std,
            eps=self.cfg.weighting.eps,
            compile_kernel=self.cfg.weighting.compile_kernel,
        )

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
            self._eigen_val_knn,
            self._eigen_vec_knn,
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
        if self.use_weighting:
            heat_weights = self.heat_weighting.compute(knn_distances)

        # Gather point evals and evecs
        grid_idx, grid_w = self.grid
        evals, evecs, _ = self.knn_gather.gather_queries(
            knn_indices,
            grid_idx,
            grid_w,
            vert_idx,
            barycentric_coords,
            self._eigen_val_knn,
            self._eigen_vec_knn,
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
        heat_qk, heat_qk_norm = self.knn_heat.query(
            indices, evals, evecs, time=diffusion_time, weights_post_diff=weights
        )
        return heat_qk, heat_qk_norm
