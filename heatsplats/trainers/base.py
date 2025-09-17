from dataclasses import dataclass, field
from abc import abstractmethod
import numpy as np
from termcolor import colored
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
    Model,
    GeodesicTracer,
    GeodesicOpt,
    EigenAlboInterpolation,
)
from heatsplats.data import MeshSamplerDataModule
from heatsplats.rendering.heat_kernels_renderer import HeatKernelsRenderer

import heatsplats.utils as utils
from heatsplats.utils import BaseObject
from heatsplats.utils.typing import *

from .utils import parse_optimizers
from heatsplats.density_controllers.utils import parse_density_controllers


class BaseTrainer(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        tracer_type: str = ""
        tracer: dict = field(default_factory=dict)

        eigen_albo: dict = field(default_factory=dict)
        model: dict = field(default_factory=dict)

        optimizers: list = field(default_factory=list)
        density_controllers: list = field(default_factory=list)

        renderer: dict = field(default_factory=dict)

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
        self.model = Model(self.cfg.model, self.mesh)

        if (
            "n_debug_traces" in self.cfg.tracer
            and self.cfg.tracer["n_debug_traces"] > self.model.N_sources
        ):
            self.cfg.tracer["n_debug_traces"] = self.model.N_sources
            heatsplats.warn(
                "Number of debug traces should not exceed number of sources. Displaying all sources instead."
            )

        self.eigalbo_interp = EigenAlboInterpolation(self.cfg.eigen_albo, self.mesh)
        self.tracer: GeodesicTracer = heatsplats.find(self.cfg.tracer_type)(
            self.cfg.tracer, self.mesh
        )

        self.optimizers = parse_optimizers(self.cfg.optimizers, self)
        self.density_controllers = parse_density_controllers(
            self.cfg.density_controllers, self.model, self.optimizers
        )

    def prepare_batch(self, data: dict) -> dict:
        for k, v in data.items():
            if isinstance(v, Tensor):
                data[k] = v.to(self.device)
        return data

    def optimise(self, n_iter=100):
        dataloader = self.datamodule.train_dataloader()
        data_iter = iter(dataloader)

        heatsplats.debug(f"INITIAL -> {self.model.colored_print_opt_params}")

        errors_lists = {k: [] for k in self.model.splat_param_keys}
        errors_lists["loss"] = []
        self.plot_model_histograms()

        for i in (pbar := tqdm(range(n_iter))):
            B = self.model.N_sources

            data = next(data_iter)
            data = self.prepare_batch(data)

            pos: Tensor = data["pos"]
            gt_colours: Tensor = data["colour"]
            evals: Tensor = data["evals"]
            pts_evecs: Tensor = data["pts_evecs"]
            pts_mass: Tensor = data["pts_mass"]
            albo_weights = data["albo_weights"]

            P = pos.shape[0]

            colours: Float[Tensor, "B P 1"] = torch.zeros([B, P, 1], device=self.device)

            kernel_vert_idx = self.mesh.get_face_vertices(self.model.kernel_face_ids)
            kernel_barycentric_coords = self.mesh.cartesian_to_barycentric(
                self.model.kernel_locations, kernel_vert_idx
            )
            self.model.save_barycentric_locations(kernel_barycentric_coords)

            kernel_evecs, kernel_mass = self.eigalbo_interp.barycentric_albo_gaussians(
                albo_weights=albo_weights,
                barycentric_coords=kernel_barycentric_coords,
                vert_idx=kernel_vert_idx,
            )

            colours: Float[Tensor, "B P+1 1"] = torch.cat(
                (colours, torch.ones([B, 1, 1], device=self.device)), dim=1
            )
            pts_evecs: Float[Tensor, "B P+1 K"] = torch.cat(
                ((pts_evecs, kernel_evecs.unsqueeze(1))), dim=1
            )
            pts_mass: Float[Tensor, "B P+1"] = torch.cat(
                (pts_mass.expand(B, -1), kernel_mass.unsqueeze(-1)), dim=1
            )

            # PS: biharmonic_dist_weights = None if 'distance_weighting' == "none"
            # in eigalbo_interp config
            biharmonic_dist_weights: Float[Tensor, "B P+1"] = (
                self.eigalbo_interp.compute_biharmonic_weights(
                    data["pts_iso_evecs"], kernel_barycentric_coords, kernel_vert_idx
                )
            )

            # pts_mass: Float[Tensor, "B P+1"] = (
            #     self.eigalbo_interp.compute_biharmonic_dist_kde_mass(
            #         data["pts_iso_evecs"],
            #         kernel_barycentric_coords,
            #         kernel_vert_idx,
            #         sigma=None,
            #         total_area_normalise=True,
            #     )
            # )

            colours = utils.heat_diffusion(
                colours,
                pts_mass,
                evals,
                pts_evecs,
                self.model.diff_times,
                biharmonic_dist_weights,
            )

            colours = colours / (
                colours[:, P, :].unsqueeze(1) + 1e-8
            )  # P is source => hottest
            colours = colours[:, :P, :]

            colours = self.model.kernel_filter_func(
                colours, epsilon=self.model.thresholds, sharpness=self.model.sharpnesses
            )

            colours = colours * self.model.opacities.view(-1, 1, 1)
            colours = colours * self.model.kernel_colours.unsqueeze(1)
            colours = colours.sum(dim=0)

            # Postprocess
            colours = self.model(colours)

            if i == 0:
                init_colours = colours.clone().detach()

            # Compute loss and backpropagate
            loss = F.mse_loss(colours, gt_colours, reduction="sum") / colours.shape[0]

            loss.backward()

            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()

            for dc in self.density_controllers:
                dc.post_backward_step(step=i)

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
        self.plot_model_histograms()
        v_colours = self.compute_vertex_colours()  # TODO: may go out of memory, batch!
        return v_colours, gt_colours, init_colours

    def compute_vertex_colours(self):
        B, V = self.model.N_sources, self.mesh.N_verts
        v_colours: Float[Tensor, "B V 1"] = torch.zeros([B, V, 1], device=self.device)

        albo_weights = self.eigalbo_interp.interpolate_anisotropies(
            angles=self.model.angles, scales=self.model.anisotropies
        )
        albo_evals, albo_evecs, mass = self.eigalbo_interp.albo_vertices(
            albo_weights=albo_weights
        )

        kernel_vert_idx = self.mesh.get_face_vertices(self.model.kernel_face_ids)
        barycentric_coords = self.mesh.cartesian_to_barycentric(
            self.model.kernel_locations, kernel_vert_idx
        )

        kernel_evecs, kernel_mass = self.eigalbo_interp.barycentric_albo_gaussians(
            albo_weights=albo_weights,
            barycentric_coords=barycentric_coords,
            vert_idx=kernel_vert_idx,
        )

        v_colours: Float[Tensor, "B P+1 1"] = torch.cat(
            (v_colours, torch.ones([B, 1, 1], device=self.device)),
            dim=1,
        )
        albo_evecs: Float[Tensor, "B V+1 K"] = torch.cat(
            ((albo_evecs, kernel_evecs.unsqueeze(1))), dim=1
        )
        mass: Float[Tensor, "B V+1"] = torch.cat(
            (mass.expand(B, -1), kernel_mass.unsqueeze(-1)), dim=1
        )

        biharmonic_dist_weights = self.eigalbo_interp.compute_biharmonic_weights(
            self.eigalbo_interp.ilbo_evec_vertices(),
            barycentric_coords,
            kernel_vert_idx,
        )

        v_colours = utils.heat_diffusion(
            v_colours,
            mass,
            albo_evals,
            albo_evecs,
            self.model.diff_times,
            biharmonic_dist_weights,
        )
        v_colours = v_colours / (
            v_colours[:, V, :].unsqueeze(1) + 1e-8
        )  # V = source => hottest
        v_colours = v_colours[:, :V, :]

        v_colours = self.model.kernel_filter_func(
            v_colours, epsilon=self.model.thresholds, sharpness=self.model.sharpnesses
        )

        v_colours = v_colours * self.model.opacities.view(-1, 1, 1)
        v_colours = v_colours * self.model.kernel_colours.unsqueeze(1)
        v_colours = v_colours.sum(dim=0)

        v_colours = self.model(v_colours)
        return v_colours

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

        renderer.mega_kernel(False)

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
        orig_opacities = self.model._opacities.clone().detach()

        # Override values
        self.model.kernel_filter_func = partial(utils.box_border, thickness=0.03)
        self.model._kernel_colours = torch.nn.Parameter(
            torch.rand_like(self.model._kernel_colours)
        )
        self.model._opacities = torch.nn.Parameter(
            torch.ones_like(self.model._opacities) * 0.5
        )

        try:
            rend_rings = self.render_result(rotating_frames)
        finally:
            # Restore original values
            self.model.kernel_filter_func = orig_kernel_filter_func
            self.model._kernel_colours = torch.nn.Parameter(orig_kernel_colours)
            self.model._opacities = torch.nn.Parameter(orig_opacities)

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
            "opacities": self.model.opacities,
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
