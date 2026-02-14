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

import heatsplats
from heatsplats.modules import (
    Mesh,
    HeatKernelTexture,
    GeodesicTracer,
    EigenAlboInterpolation,
    KernelInfo,
    PointsInfo,
)
from heatsplats.data import MeshSamplerDataModule
from heatsplats.rendering.heat_kernels_renderer import HeatKernelsRenderer

import heatsplats.utils as utils
from heatsplats.utils import BaseObject
from heatsplats.utils.typing import *

from .utils import parse_optimizers_and_schedulers
from heatsplats.density_controllers.utils import parse_density_controllers


class BaseTrainer(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        tracer_type: str = ""
        tracer: dict = field(default_factory=dict)

        eigen_albo_type: str = "modules.eigen-albo-interpolation"
        eigen_albo: dict = field(default_factory=dict)
        model: dict = field(default_factory=dict)

        loss_type: str = "mse_loss"  # any torch.nn.functional (e.g., smooth_l1_loss)

        optimizers: list = field(default_factory=list)
        density_controllers: list = field(default_factory=list)

        renderer: dict = field(default_factory=dict)
        renderer_mega_kernel: bool = False

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
        self.model = HeatKernelTexture(self.cfg.model, self.mesh)

        if (
            "n_debug_traces" in self.cfg.tracer
            and self.cfg.tracer["n_debug_traces"] > self.model.N_sources
        ):
            self.cfg.tracer["n_debug_traces"] = self.model.N_sources
            heatsplats.warn(
                "Number of debug traces should not exceed number of sources. Displaying all sources instead."
            )

        EigenAlboClass = heatsplats.find(self.cfg.eigen_albo_type)
        self.eigalbo_interp: EigenAlboInterpolation = EigenAlboClass(
            self.cfg.eigen_albo, self.mesh
        )
        self.tracer: GeodesicTracer = heatsplats.find(self.cfg.tracer_type)(
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

    def optimise(self, n_iter=100):
        dataloader = self.datamodule.train_dataloader()
        data_iter = iter(dataloader)

        heatsplats.debug(f"INITIAL -> {self.model.colored_print_opt_params}")

        errors_lists = {k: [] for k in self.model.splat_param_keys}
        errors_lists["loss"] = []
        self.plot_model_histograms()

        grads_lists = {k: [] for k, _ in self.model.named_parameters()}

        for i in (pbar := tqdm(range(n_iter))):

            data = next(data_iter)
            data = self.prepare_batch(data)

            gt_colours: Tensor = data["colour"]
            points_info: PointsInfo = data["points_info"]
            albo_weights = data["albo_weights"]

            with torch.profiler.record_function("prepare_kernels_for_diffusion"):
                kernel_info: KernelInfo = self.model.prepare_kernels_for_diffusion(
                    mesh=self.mesh,
                    eigalbo_interp=self.eigalbo_interp,
                    albo_weights=albo_weights,
                )

            with torch.profiler.record_function("diffuse_heat_kernels"):
                colours, kernel_contributions = self.model.diffuse_heat_kernels(
                    eigalbo_interp=self.eigalbo_interp,
                    pts_info=points_info,
                    kernel_info=kernel_info,
                    at_vertices=False,
                )  # [P, D]

            colours = self.model(colours)  # Postprocess

            if i == 0:
                init_colours = colours.clone().detach()

            # Compute loss, backpropagate, and update all other objects
            per_point_loss = self.loss_func(colours, gt_colours, reduction="none").sum(
                dim=1
            )
            loss = per_point_loss.sum() / colours.shape[0]

            for dc in self.density_controllers:
                dc.pre_backward_step(
                    step=i,
                    rendered_colours=colours,
                    gt_colours=gt_colours,
                    kernel_contributions=kernel_contributions,
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

            if heatsplats.is_debug() and (i == 0 or (i + 1) % 100 == 0):
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grads_lists[name].append(param.grad.norm().item())

            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()

            for scheduler in self.schedulers:
                scheduler.step()

            self.model.post_optimizer_step()

            if self.datamodule.cfg.use_importance_sampling:
                self.datamodule.update_errors(
                    data["pool_indices"], per_point_loss.detach()
                )

            with torch.no_grad():
                errors = self._errors
                loss_step = loss.item()
                if i == 0 or (i + 1) % 100 == 0:
                    heatsplats.debug(
                        f"Iteration: {i + 1} -> Loss: {loss_step}. {errors['printables']}",
                    )

                for k in self.model.splat_param_keys:
                    if k in errors:
                        errors_lists[k].append(errors[k].item())
                errors_lists["loss"].append(loss_step)
                pbar.set_postfix_str(f"Loss: {loss_step:0.4f}")

        heatsplats.debug(f"FINAL -> {self.model.colored_print_opt_params}")

        self.plot_errors(errors_lists)

        if heatsplats.is_debug():
            self.plot_model_histograms()
            self.plot_gradient_norms(grads_lists, log_interval=100, y_log_scale=True)

        v_colours = None
        # TODO: may go out of memory, batch!
        # v_colours = self.model.compute_vertex_colours()
        return v_colours, gt_colours, init_colours

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
        self, rotating_frames: int = 10
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
        renderer = HeatKernelsRenderer(self.cfg.renderer)

        renderer.mega_kernel(
            self.cfg.renderer_mega_kernel, no_loops=True, no_opt_calls=True
        )

        mi_mesh = renderer.mesh_to_mitsuba(
            self.datamodule.mesh, self.mesh, self.model, self.eigalbo_interp
        )

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

        # Override values
        self.model.kernel_filter_func = partial(utils.box_border, thickness=0.03)
        self.model._kernel_colours = torch.nn.Parameter(
            torch.rand_like(self.model._kernel_colours)
        )

        try:
            rend_rings = self.render_result(rotating_frames)
        finally:
            # Restore original values
            self.model.kernel_filter_func = orig_kernel_filter_func
            self.model._kernel_colours = torch.nn.Parameter(orig_kernel_colours)

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
