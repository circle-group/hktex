from dataclasses import dataclass

import matplotlib.pyplot as plt
import mitsuba as mi
import torch

import hktex
from hktex.data import MeshSamplerDataModule
from hktex.utils import total_variation_loss, smoothness_loss
from hktex.utils.typing import *

from .base import BaseTrainer
from .uv_texture import UvTextureTrainer


@hktex.register("trainers.uv-texture-pcl")
class UvTexturePCLTrainer(UvTextureTrainer):
    @dataclass
    class Config(UvTextureTrainer.Config):
        tv_loss_weight: float = 0.0
        smoothness_loss_weight: float = 0.0
        edge_aware_reg_enabled: bool = False
        edge_aware_reg_sigma_colour: float = 0.1
        edge_aware_reg_min_weight: float = 0.05
        edge_aware_reg_eps: float = 1e-8
        edge_aware_reg_detach_colours: bool = True
        edge_aware_photo_enabled: bool = False
        edge_aware_photo_k: int = 8
        edge_aware_photo_strength: float = 1.0
        edge_aware_photo_max_weight: float = 4.0
        edge_aware_photo_eps: float = 1e-8

    cfg: Config

    def configure(
        self,
        datamodule: MeshSamplerDataModule,
        **kwargs,
    ):
        super().configure(datamodule, **kwargs)
        if not self.cfg.use_knn_implementation:
            raise ValueError("UvTexturePCLTrainer expects use_knn_implementation=True.")
        required = ("prepare_points", "prepare_kernels", "diffuse_heat_kernels")
        for name in required:
            if not hasattr(self.model, name):
                raise TypeError(
                    f"Configured model is missing required method '{name}' for PCL KNN flow."
                )
        zero = torch.tensor(0.0, device=self.device)
        self._loss_terms = {
            "photo_loss": zero,
            "tv_loss": zero,
            "smoothness_loss": zero,
        }

    def prepare_batch(self, data: dict) -> dict:
        data = BaseTrainer.prepare_batch(self, data)
        face_ids = data["face_id"]
        barys = data["bary"]

        points_info = self.model.prepare_points(
            mesh=self.mesh,
            face_ids=face_ids,
            barys=barys,
            pts=None,
        )
        data["points_info"] = points_info
        return data

    def prepare_knn(self, save_barycentric=True):
        return self.model.prepare_kernels(
            mesh=self.mesh, save_barycentric=save_barycentric
        )

    def _compute_edge_aware_photo_loss(
        self,
        per_point_loss: Tensor,
        gt_colours: Tensor,
        data: Optional[dict],
    ) -> Tensor:
        if (
            not self.cfg.edge_aware_photo_enabled
            or data is None
            or "points_info" not in data
            or "pts" not in data["points_info"]
        ):
            return per_point_loss.mean()

        pts = data["points_info"]["pts"]
        P = int(pts.shape[0])
        k = int(max(1, min(self.cfg.edge_aware_photo_k, P - 1)))
        if k <= 0:
            return per_point_loss.mean()

        nn_idx = self._compute_knn_indices(pts, k)
        edge_score = self._compute_edge_score(gt_colours, nn_idx)

        edge_norm = edge_score / (
            edge_score.mean() + float(self.cfg.edge_aware_photo_eps)
        )
        weights = 1.0 + float(self.cfg.edge_aware_photo_strength) * edge_norm
        weights = weights.clamp(max=float(self.cfg.edge_aware_photo_max_weight))
        return (weights * per_point_loss).sum() / (
            weights.sum() + float(self.cfg.edge_aware_photo_eps)
        )

    def _compute_knn_indices(self, pts: Tensor, k: int) -> Tensor:
        d2 = torch.cdist(pts, pts, p=2)
        _, nn_idx = torch.topk(d2, k=k + 1, largest=False, dim=1)
        return nn_idx[:, 1:]

    def _compute_pair_deltas(self, values: Tensor, nn_indices: Tensor) -> Tensor:
        center = values.unsqueeze(1)
        neigh = values[nn_indices]
        return neigh - center

    def _compute_edge_score(self, values: Tensor, nn_indices: Tensor) -> Tensor:
        delta = self._compute_pair_deltas(values, nn_indices)
        return delta.abs().mean(dim=(1, 2))

    def _compute_edge_aware_pair_weights(
        self,
        kernel_colours: Tensor,
        nn_indices: Tensor,
        nn_dists: Tensor,
    ) -> Optional[Tensor]:
        if not self.cfg.edge_aware_reg_enabled:
            return None

        source = (
            kernel_colours.detach()
            if self.cfg.edge_aware_reg_detach_colours
            else kernel_colours
        )
        delta = self._compute_pair_deltas(source, nn_indices)
        colour_delta = delta.pow(2).sum(dim=-1).sqrt()

        sigma_c = max(float(self.cfg.edge_aware_reg_sigma_colour), 1e-8)
        edge_gate = torch.exp(-colour_delta / sigma_c)
        edge_gate = edge_gate.clamp(min=float(self.cfg.edge_aware_reg_min_weight))

        # Keep the same spatial weighting style as the default regularizers.
        sigma_d = 0.05
        spatial_w = torch.exp(-nn_dists / (2.0 * sigma_d * sigma_d))
        return spatial_w * edge_gate

    def _graph_regulariser_with_pair_weights(
        self,
        kernel_colours: Tensor,
        nn_indices: Tensor,
        pair_weights: Tensor,
        mode: str,
    ) -> Tensor:
        delta = self._compute_pair_deltas(kernel_colours, nn_indices)

        if mode == "tv":
            pair_term = delta.abs().sum(dim=-1)
        elif mode == "smoothness":
            pair_term = delta.pow(2).sum(dim=-1)
        else:
            raise ValueError(f"Unknown regulariser mode: {mode}")

        eps = float(self.cfg.edge_aware_reg_eps)
        per_node = (pair_term * pair_weights).sum(dim=1) / (
            pair_weights.sum(dim=1) + eps
        )
        return per_node.mean()

    def _compute_graph_regulariser_raw(
        self,
        kernel_colours: Tensor,
        nn_indices: Tensor,
        nn_dists: Tensor,
        pair_weights: Optional[Tensor],
        mode: str,
    ) -> Tensor:
        if pair_weights is not None:
            return self._graph_regulariser_with_pair_weights(
                kernel_colours=kernel_colours,
                nn_indices=nn_indices,
                pair_weights=pair_weights,
                mode=mode,
            )
        if mode == "tv":
            return total_variation_loss(
                kernel_colours,
                nn_indices,
                nn_distances=nn_dists,
            )
        if mode == "smoothness":
            return smoothness_loss(
                kernel_colours,
                nn_indices,
                nn_distances=nn_dists,
            )
        raise ValueError(f"Unknown regulariser mode: {mode}")

    def compute_loss(
        self,
        colours: Tensor,
        gt_colours: Tensor,
        data: Optional[dict] = None,
    ) -> tuple[Tensor, Tensor]:
        per_point_loss = self.loss_func(colours, gt_colours, reduction="none").sum(
            dim=1
        )
        loss = self._compute_edge_aware_photo_loss(per_point_loss, gt_colours, data)

        self._loss_terms["photo_loss"] = loss.detach()
        self._loss_terms["tv_loss"] = torch.tensor(0.0, device=loss.device)
        self._loss_terms["smoothness_loss"] = torch.tensor(0.0, device=loss.device)

        tv_w = float(self.cfg.tv_loss_weight)
        sm_w = float(self.cfg.smoothness_loss_weight)
        if tv_w <= 0.0 and sm_w <= 0.0:
            return loss, per_point_loss

        G = int(self.model.N_sources)
        if G < 2:
            return loss, per_point_loss

        k = int(max(1, min(self.model.cfg.knn_k, G - 1)))
        db = (self.model.kernel_locations + 0.0).contiguous()

        with torch.no_grad():
            nn_dists, nn_indices = self.model._faiss_index.search(db, k=k + 1)
            nn_indices = nn_indices[:, 1:]
            nn_dists = nn_dists[:, 1:]

        kernel_colours = self.model.kernel_colours
        pair_weights = self._compute_edge_aware_pair_weights(
            kernel_colours=kernel_colours,
            nn_indices=nn_indices,
            nn_dists=nn_dists,
        )
        if tv_w > 0.0:
            tv_loss_raw = self._compute_graph_regulariser_raw(
                kernel_colours=kernel_colours,
                nn_indices=nn_indices,
                nn_dists=nn_dists,
                pair_weights=pair_weights,
                mode="tv",
            )
            tv_loss = tv_w * tv_loss_raw
            self._loss_terms["tv_loss"] = tv_loss.detach()
            loss = loss + tv_loss
        if sm_w > 0.0:
            sm_loss_raw = self._compute_graph_regulariser_raw(
                kernel_colours=kernel_colours,
                nn_indices=nn_indices,
                nn_dists=nn_dists,
                pair_weights=pair_weights,
                mode="smoothness",
            )
            sm_loss = sm_w * sm_loss_raw
            self._loss_terms["smoothness_loss"] = sm_loss.detach()
            loss = loss + sm_loss

        return loss, per_point_loss

    def extra_error_keys(self) -> list[str]:
        return ["photo_loss", "tv_loss", "smoothness_loss"]

    def reset_knn(self):
        self.model.reset()

    @property
    def _errors(self):
        return {
            "printables": None,
            "kernel_colours": self.model.kernel_colours.abs().mean(),
            "softmax_temperature": self.model.softmax_temperature.mean(),
            "residual_gain": self.model.residual_gain.mean(),
            "photo_loss": self._loss_terms["photo_loss"],
            "tv_loss": self._loss_terms["tv_loss"],
            "smoothness_loss": self._loss_terms["smoothness_loss"],
        }

    def plot_model_histograms(self):
        props = {
            "kernel_colours": self.model.kernel_colours,
        }

        plt.figure(figsize=(6, 4))
        for i, (name, tensor) in enumerate(props.items(), 1):
            plt.subplot(1, 1, i)
            arr = tensor.detach().cpu().reshape(-1).numpy()
            plt.hist(arr, bins=30)
            plt.title(name)
            plt.xlabel("Value")
            plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()
        print(f"softmax_temperature: {self.model.softmax_temperature.item():.6f}")
        print(f"residual_gain: {self.model.residual_gain.item():.6f}")

    def render_kernel_rings(
        self, rotating_frames: int = 10, thickness: float = 0.03
    ) -> Union[mi.Bitmap, list[mi.Bitmap]]:
        # PCLTexture has no kernel_filter_func. Render a similar diagnostic by
        # visualizing sharp random kernel influence regions.
        orig_kernel_colours = self.model._kernel_colours.clone().detach()
        orig_mean_colour = self.model._mean_colour.clone().detach()
        orig_tau = self.model._softmax_temperature_raw.clone().detach()

        if self.model.cfg.allow_negative_colours:
            colour_sample = torch.rand_like(self.model._kernel_colours) * 2.0 - 1.0
        else:
            colour_sample = torch.rand_like(self.model._kernel_colours).clamp(1e-6, 1.0)

        if self.model.cfg.weighting == "softmax_rbf":
            tau_val = torch.tensor([0.005], device=self.device, dtype=torch.float)
            tau_raw = self.model._inv_tau_act(tau_val)
        else:
            tau_raw = orig_tau

        self.model._kernel_colours = torch.nn.Parameter(
            self.model._inv_colour_act(colour_sample)
        )
        self.model._mean_colour = torch.nn.Parameter(
            torch.zeros_like(self.model._mean_colour)
        )
        self.model._softmax_temperature_raw = torch.nn.Parameter(tau_raw)

        try:
            rend_regions = self.render_result(rotating_frames)
        finally:
            self.model._kernel_colours = torch.nn.Parameter(orig_kernel_colours)
            self.model._mean_colour = torch.nn.Parameter(orig_mean_colour)
            self.model._softmax_temperature_raw = torch.nn.Parameter(orig_tau)

        return rend_regions
