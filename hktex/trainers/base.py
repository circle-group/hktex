from dataclasses import dataclass, field
from abc import abstractmethod
import numpy as np
from termcolor import colored
import os
from functools import partial
import trimesh
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import mitsuba as mi

import hktex
from hktex.modules import (
    Mesh,
    HeatKernelTexture,
    HeatKernelTextureKNN,
    GeodesicTracer,
    EigenAlboInterpolation,
    KernelInfo,
    PointsInfo,
)
from hktex.data import MeshSamplerDataModule
from hktex.rendering.heat_kernels_renderer import HeatKernelsRenderer
from hktex.rendering.heat_kernels_renderer_knn import HeatKernelsRendererKNN

import hktex.utils as utils
from hktex.utils import BaseObject
from hktex.utils.video import save_video
from hktex.utils.typing import *

from .utils import parse_optimizers_and_schedulers
from hktex.density_controllers.utils import parse_density_controllers


class BaseTrainer(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        tracer_type: str = ""
        tracer: dict = field(default_factory=dict)

        eigen_albo_type: str = "modules.eigen-albo-interpolation"
        eigen_albo: dict = field(default_factory=dict)

        model_type: Optional[str] = None
        model: dict = field(default_factory=dict)

        loss_type: str = "mse_loss"  # any torch.nn.functional (e.g., smooth_l1_loss)
        data_initialisation_random_ratio: float = 0.3

        optimizers: list = field(default_factory=list)
        density_controllers: list = field(default_factory=list)

        renderer: dict = field(default_factory=dict)
        renderer_mega_kernel: bool = False

        use_knn_implementation: bool = False
        debug_video_frequency: int = 500

    cfg: Config

    def configure(
        self,
        datamodule: MeshSamplerDataModule,
        **kwargs,
    ):
        super().configure()

        if "renderer_cfg" in kwargs:
            self.cfg.renderer = kwargs["renderer_cfg"]

        self.datamodule = datamodule

        self.mesh = Mesh.from_trimesh(self.datamodule.mesh, device=self.device)
        if self.cfg.model_type is not None:
            ModelClass = hktex.find(self.cfg.model_type)
            self.model = ModelClass(self.cfg.model, self.mesh)
        else:
            if self.cfg.use_knn_implementation:
                self.model = HeatKernelTextureKNN(self.cfg.model, self.mesh)
            else:
                self.model = HeatKernelTexture(self.cfg.model, self.mesh)

        if (
            "n_debug_traces" in self.cfg.tracer
            and self.cfg.tracer["n_debug_traces"] > self.model.N_sources
        ):
            self.cfg.tracer["n_debug_traces"] = self.model.N_sources
            hktex.warn(
                "Number of debug traces should not exceed number of sources. Displaying all sources instead."
            )

        self.eigalbo_interp: Optional[EigenAlboInterpolation] = None
        if self.cfg.eigen_albo_type:
            EigenAlboClass = hktex.find(self.cfg.eigen_albo_type)
            self.eigalbo_interp = EigenAlboClass(self.cfg.eigen_albo, self.mesh)
        self.tracer: GeodesicTracer = hktex.find(self.cfg.tracer_type)(
            self.cfg.tracer, self.mesh
        )

        self.loss_func = getattr(F, self.cfg.loss_type)
        self.optimizers, self.schedulers = parse_optimizers_and_schedulers(
            self.cfg.optimizers, self
        )
        self.density_controllers = parse_density_controllers(
            self.cfg.density_controllers, self.mesh, self.model, self.optimizers
        )

    def prepare_batch(self, data: dict) -> dict:
        return self._move_to_device(data)

    def _move_to_device(self, obj):
        if isinstance(obj, Tensor):
            return obj.to(self.device)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        else:
            return obj

    @abstractmethod
    def data_dependent_initialisation(self, **kwargs):
        raise NotImplementedError

    def compute_loss(
        self,
        colours: Tensor,
        gt_colours: Tensor,
        data: Optional[dict] = None,
    ) -> tuple[Tensor, Tensor]:
        per_point_loss = self.loss_func(colours, gt_colours, reduction="none").sum(
            dim=1
        )
        loss = per_point_loss.sum() / colours.shape[0]
        return loss, per_point_loss

    def extra_error_keys(self) -> list[str]:
        return []

    def optimise(self, n_iter=100, debug_log_dir=None):
        dataloader = self.datamodule.train_dataloader()
        data_iter = iter(dataloader)

        random_ratio = self.cfg.data_initialisation_random_ratio
        if random_ratio < 1.0:
            self.data_dependent_initialisation(random_ratio=random_ratio)

        hktex.debug(f"INITIAL -> {self.model.colored_print_opt_params}")

        tracked_error_keys = list(self.model.splat_param_keys) + self.extra_error_keys()
        errors_lists = {k: [] for k in tracked_error_keys}
        errors_lists["loss"] = []
        self.plot_model_histograms()

        grads_lists = {k: [] for k, _ in self.model.named_parameters()}

        for i in (pbar := tqdm(range(n_iter))):
            if debug_log_dir is not None and (
                i == 0 or i % self.cfg.debug_video_frequency == 0
            ):
                current_rnd = self.render_result(self.cfg.renderer.n_rotating_frames)
                save_video(current_rnd, os.path.join(debug_log_dir, f"iter_{i}.mp4"))

            data = next(data_iter)
            if self.cfg.use_knn_implementation:
                kernel_info = self.prepare_knn()
            data = self.prepare_batch(data)

            gt_colours: Tensor = data["colour"]

            if self.cfg.use_knn_implementation:
                (
                    colours,
                    kernel_contributions,
                    topk_kernel_idxs,
                    topk_kernel_contribs,
                ) = self.forward_knn(data)
            else:
                colours, kernel_contributions, topk_kernel_idxs, kernel_info = (
                    self.forward_model(data)
                )
                topk_kernel_contribs = None

            if i == 0:
                init_colours = colours.clone().detach()

            # Compute loss, backpropagate, and update all other objects
            loss, per_point_loss = self.compute_loss(colours, gt_colours, data=data)

            for dc in self.density_controllers:
                dc.pre_backward_step(
                    step=i,
                    rendered_colours=colours,
                    gt_colours=gt_colours,
                    kernel_contributions=kernel_contributions,
                    topk_kernel_idxs=topk_kernel_idxs,
                    topk_kernel_contribs=topk_kernel_contribs,
                )

            loss.backward()

            for dc in self.density_controllers:
                dc.refresh_state()
                dc.post_backward_step(
                    step=i,
                    eigalbo_interp=self.eigalbo_interp,
                    kernel_info=kernel_info,
                    tracer=self.tracer,
                )

            if hktex.is_debug() and (i == 0 or (i + 1) % 100 == 0):
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grads_lists[name].append(param.grad.norm().item())

            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()

            for scheduler in self.schedulers:
                scheduler.step()

            self.model.post_optimizer_step()
            if self.cfg.use_knn_implementation:
                self.mark_knn_dirty()

            if self.datamodule.cfg.use_importance_sampling:
                self.datamodule.update_errors(
                    data["pool_indices"], per_point_loss.detach()
                )

            with torch.no_grad():
                errors = self._errors
                loss_step = loss.item()
                if i == 0 or (i + 1) % 100 == 0:
                    hktex.debug(
                        f"Iteration: {i + 1} -> Loss: {loss_step}. {errors['printables']}",
                    )

                for k in tracked_error_keys:
                    if k in errors:
                        errors_lists[k].append(errors[k].item())
                errors_lists["loss"].append(loss_step)
                pbar.set_postfix_str(f"Loss: {loss_step:0.4f}")

            if self.cfg.use_knn_implementation:
                self.reset_knn()

        hktex.debug(f"FINAL -> {self.model.colored_print_opt_params}")

        self.plot_errors(errors_lists)

        if hktex.is_debug():
            self.plot_model_histograms()
            self.plot_gradient_norms(grads_lists, log_interval=100, y_log_scale=True)

        v_colours = None
        # TODO: may go out of memory, batch!
        # v_colours = self.model.compute_vertex_colours()
        return v_colours, gt_colours, init_colours

    def forward_model(self, data):
        points_info: PointsInfo = data["points_info"]
        albo_weights = data["albo_weights"]

        with torch.profiler.record_function("prepare_kernels_for_diffusion"):
            kernel_info: KernelInfo = self.model.prepare_kernels_for_diffusion(
                mesh=self.mesh,
                eigalbo_interp=self.eigalbo_interp,
                albo_weights=albo_weights,
            )

        with torch.profiler.record_function("diffuse_heat_kernels"):
            colours, kernel_contributions, topk_kernel_idxs = (
                self.model.diffuse_heat_kernels(
                    eigalbo_interp=self.eigalbo_interp,
                    pts_info=points_info,
                    kernel_info=kernel_info,
                    at_vertices=False,
                )
            )  # [P, D]

        colours = self.model(colours)  # Postprocess
        return colours, kernel_contributions, topk_kernel_idxs, kernel_info

    def prepare_knn(self, save_barycentric=True):
        self.model: HeatKernelTextureKNN
        return self.model.prepare_kernels(
            self.mesh, self.eigalbo_interp, save_barycentric=save_barycentric
        )

    def reset_knn(self):
        self.model.reset(self.eigalbo_interp)

    def mark_knn_dirty(self):
        if hasattr(self.model, "mark_knn_dirty"):
            self.model.mark_knn_dirty()

    def forward_knn(self, data):
        self.model: HeatKernelTextureKNN
        points_info: PointsInfo = data["points_info"]

        colours, kernel_contributions, topk_kernel_idxs, topk_kernel_contribs = (
            self.model.diffuse_heat_kernels(
                eigalbo_interp=self.eigalbo_interp, pts_info=points_info
            )
        )  # [P, D]

        colours = self.model(colours)  # Postprocess
        return colours, kernel_contributions, topk_kernel_idxs, topk_kernel_contribs

    @abstractmethod
    def render_gt(self, rotating_frames: int = 10) -> Union[mi.Bitmap, list[mi.Bitmap]]:
        """
        Render the ground truth mesh. Defined in subclases as the GT mesh could have
        vertex_colours, uv_textures, or nothing (if texture comes from images).

        Args:
            rotating_frames (int): Number of frames for rotation.
                If 1, render a single image.
        Returns:
            mi.Bitmap or list[mi.Bitmap]: The rendered image(s).
        """
        pass

    def render_result(
        self, rotating_frames: int = 10, **kwargs
    ) -> Union[mi.Bitmap, list[mi.Bitmap]]:
        """
        Render the mesh with the resultsing heat kernel texture. This is always rendered
        with the heat kernel texture, so it is not defined in subclasses.
        Args:
            rotating_frames (int): Number of frames for rotation.
                If 1, render a single image.
        Returns:
            mi.Bitmap or list[mi.Bitmap]: The rendered image(s).
        """
        if self.cfg.use_knn_implementation:
            renderer = HeatKernelsRendererKNN(self.cfg.renderer)
        else:
            renderer = HeatKernelsRenderer(self.cfg.renderer)

        renderer.mega_kernel(
            self.cfg.renderer_mega_kernel, no_loops=True, no_opt_calls=True
        )

        mi_mesh = renderer.mesh_to_mitsuba(
            self.datamodule.mesh, self.mesh, self.model, self.eigalbo_interp, **kwargs
        )

        if self.cfg.use_knn_implementation:
            self.prepare_knn(False)
        if rotating_frames == 1:
            img = renderer.render(mi_mesh, denoise=True)
            out = mi.Bitmap(img).convert(
                pixel_format=mi.Bitmap.PixelFormat.RGB,
                component_format=mi.Struct.Type.UInt8,
                srgb_gamma=True,
            )
        else:
            out = renderer.rotating_video(mi_mesh, rotating_frames)

        renderer.flush_cache()
        if self.cfg.use_knn_implementation:
            self.reset_knn()

        return out

    def render_kernel_rings(
        self, rotating_frames: int = 10, thickness: float = 0.03
    ) -> Union[mi.Bitmap, list[mi.Bitmap]]:
        """
        Render the mesh with the contour of the resultsing heat kernels.
        This is always rendered with the heat kernel texture, so it is not defined
        in subclasses.
        Args:
            rotating_frames (int): Number of frames for rotation.
                If 1, render a single image.
            thickness (float): Thickness of the contour lines.
        Returns:
            mi.Bitmap or list[mi.Bitmap]: The rendered image(s).
        """
        # Save original values
        orig_kernel_filter_func = self.model.kernel_filter_func
        orig_kernel_colours = self.model._kernel_colours.clone().detach()
        orig_mean_colour = self.model._mean_colour.clone().detach()

        # Override values
        self.model.kernel_filter_func = partial(utils.box_border, thickness=0.03)
        self.model._kernel_colours = torch.nn.Parameter(
            torch.rand_like(self.model._kernel_colours)
        )
        self.model._mean_colour = torch.nn.Parameter(
            torch.zeros_like(self.model._mean_colour)
        )

        try:
            rend_rings = self.render_result(rotating_frames)
        finally:
            # Restore original values
            self.model.kernel_filter_func = orig_kernel_filter_func
            self.model._kernel_colours = torch.nn.Parameter(orig_kernel_colours)
            self.model._mean_colour = torch.nn.Parameter(orig_mean_colour)

        return rend_rings

    @property
    def kernel_centres(self):
        return self.model.kernel_locations

    @property
    def _errors(self):
        return {
            "printables": None,
            "angles": torch.tensor(0),
            "anisotropies": torch.tensor(0),
            "diff_times": torch.tensor(0),
            "kernel_colours": torch.tensor(0),
        }

    @staticmethod
    def plot_errors(errors_lists):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].plot(errors_lists["loss"], label="Loss")
        axes[0].set_title("Loss per step")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(errors_lists["loss"], label="Loss")
        axes[1].set_yscale("log")  # Set y-axis to log scale
        axes[1].set_title("Loss per Step (Log Scale)")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Loss (Log Scale)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        extra_keys = [k for k in errors_lists.keys() if k != "loss"]
        plotted = False
        for k in extra_keys:
            if len(errors_lists[k]) > 0:
                plotted = True
                break

        if plotted:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            for k in extra_keys:
                vals = errors_lists[k]
                if len(vals) > 0:
                    axes[0].plot(vals, label=k)
                    axes[1].plot(vals, label=k)
            axes[0].set_title("Tracked Error Terms")
            axes[0].set_xlabel("Iteration")
            axes[0].set_ylabel("Value")
            axes[0].legend()

            axes[1].set_yscale("log")
            axes[1].set_title("Tracked Error Terms (Log Scale)")
            axes[1].set_xlabel("Iteration")
            axes[1].set_ylabel("Value (Log Scale)")
            axes[1].legend()
            plt.tight_layout()
            plt.show()

    def plot_model_histograms(self):
        props = {
            "sharpnesses": self.model.sharpnesses,
            "thresholds": self.model.thresholds,
            "anisotropies": self.model.anisotropies,
            "angles (deg)": self.model.angles * 180 / torch.pi,
        }

        plt.figure(figsize=(15, 8))
        for i, (name, tensor) in enumerate(props.items(), 1):
            plt.subplot(2, 3, i)
            arr = tensor.detach().cpu().numpy()
            plt.hist(arr, bins=30)
            plt.title(name)
            plt.xlabel("Value")
            plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def plot_gradient_norms(
        self,
        grads_lists: dict[str, list[float]],
        log_interval: int = 100,
        y_log_scale: bool = True,
    ):
        if not grads_lists:
            print("Gradient dictionary is empty. Nothing to plot.")
            return

        # Generate the x-axis values based on the logging frequency
        num_logs = 0
        for name, norms in grads_lists.items():
            if norms:  # Find the first non-empty list to get the length
                num_logs = len(norms)
                break

        if num_logs == 0:
            print("All gradient lists are empty. Nothing to plot.")
            return

        steps = [0] + [(i * log_interval) - 1 for i in range(1, num_logs)]

        fig, ax = plt.subplots(figsize=(15, 8))
        for name, norm_list in grads_lists.items():
            if not norm_list:
                hktex.warn(f"Gradient list for {name} is empty. Skipping.")
                continue
            ax.plot(steps, norm_list, label=name[1:], markersize=4, marker="o")

        ax.set_xlabel("Training Step")
        ax.set_ylabel("L2 Norm of Gradient")
        ax.set_title("Gradient Norms During Training")

        if y_log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("L2 Norm of Gradient (Log Scale)")

        ax.legend(framealpha=0.7, loc="best")
        plt.show()

    @property
    def debug_trimesh_traces(self):
        assert self.tracer.debug, "Tracer not in debug mode"
        traces_info = self.tracer.full_traces_info
        # segment_starts = np.concatenate(traces_info["starts"], axis=0)
        traces_starts = np.stack([s[0, ::] for s in traces_info["starts"]])
        traces = traces_info["traces"]
        return [
            utils.big_trimesh_pcl(traces_starts, None, radius=0.005),
            *[trimesh.load_path(t, colors=[[255, 0, 0, 255]]) for t in traces],
        ]

    def save_model(self, filename):
        self.model.save_torch(filename)
        torch_size = os.path.getsize(filename)

        npz_filename = filename.replace(".pt", ".npz")
        self.model.save_numpy_npz(npz_filename)
        npz_size = os.path.getsize(npz_filename)
        return torch_size / 1024, npz_size / 1024
