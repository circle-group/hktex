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
import drjit as dr

import heatsplats
from heatsplats.modules import (
    Mesh,
)
from heatsplats.modules.base import TextureNetwork
from heatsplats.data import MeshSamplerDataModule
from heatsplats.rendering.torch_texture_renderer import TorchTextureRenderer
from heatsplats.rendering.uv_texture_renderer import UVTextureRenderer

import heatsplats.utils as utils
from heatsplats.utils import ObjectWithCallbacks, load_mesh
from heatsplats.utils.typing import *

from .utils import parse_optimizers_and_schedulers


class MitsubaTrainer(ObjectWithCallbacks):
    @dataclass
    class Config(ObjectWithCallbacks.Config):
        network_type: str = "modules.mlp-texture-network"
        network: dict = field(default_factory=dict)

        optimizers: list = field(default_factory=list)

        renderer_type: str = "renderer.torch-texture"
        renderer: dict = field(default_factory=dict)
        renderer_mega_kernel: bool = False

        denoise_ad_prop: bool = False
        spp: int = 0

        loss_type: str = "mse_loss"
        loss_force_vectorized: bool = True

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
        self.model: TextureNetwork = heatsplats.find(self.cfg.network_type)(
            self.cfg.network, mesh=self.mesh
        )
        self.loss_fn = utils.mitsuba.get_mitsuba_loss(
            self.cfg.loss_type, self.cfg.loss_force_vectorized
        )

        self.optimizers, self.schedulers = parse_optimizers_and_schedulers(
            self.cfg.optimizers, self.model
        )

        self.renderer = self._get_renderer()

    def _get_renderer(self):
        renderer: TorchTextureRenderer = heatsplats.find(self.cfg.renderer_type)(
            self.cfg.renderer
        )
        renderer.mega_kernel(
            self.cfg.renderer_mega_kernel, no_loops=True, no_opt_calls=True
        )
        return renderer

    def prepare_batch(self, data: dict) -> dict:
        batch_cameras = [dict() for _ in range(data["batch_size"])]
        for k, v in data["cameras"].items():
            for i, c_v in enumerate(v):
                batch_cameras[i][k] = c_v.item()

        return batch_cameras

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

        if self.model.requires_scene_bounds():
            mesh_texless = self.renderer.mesh_notex_to_mitsuba(self.datamodule.mesh)
            scene_texless = self.renderer.make_scene(mesh_texless, False)
            scene_min, scene_max = scene_texless.bbox().min, scene_texless.bbox().max
            self.model.set_scene_bounds(scene_min.torch(), scene_max.torch())

        mi_mesh, mi_texture = self.renderer.mesh_to_mitsuba(
            self.datamodule.mesh, self.model
        )
        scene, params = self.renderer.make_scene(mi_mesh, with_params=True)

        grad_params_dict = utils.TraversableDict(
            {"texture": [mi_texture, mi.ParamFlags.Differentiable]}
        )
        grad_params = mi.traverse(grad_params_dict)
        dr.enable_grad(grad_params["texture.grad_activator"])
        print(grad_params)

        seed = 0

        errors_lists = {"loss": []}
        grads_lists = {k: [] for k, _ in self.model.named_parameters()}

        for i in (pbar := tqdm(range(n_iter))):
            data = next(data_iter)
            batch_cameras = self.prepare_batch(data)
            batch_size = len(batch_cameras)

            with torch.no_grad():
                with dr.suspend_grad():
                    gt_images = self._render_gt_tiles(batch_cameras, seed)

            for optimizer in self.optimizers:
                self.foreach_callback(
                    lambda cb: cb.on_before_zero_grad(self, optimizer)
                )
                optimizer.zero_grad()

            total_loss = 0.0
            for c_i, camera_params in enumerate(batch_cameras):
                backward_success = False
                while not backward_success:
                    self.renderer.update_camera_param(params, **camera_params)

                    img_i = mi.render(
                        scene,
                        spp=self.cfg.spp,
                        params=grad_params,
                        seed=seed + c_i,
                        seed_grad=seed + c_i + 1,
                    )
                    if self.cfg.denoise_ad_prop:
                        denoiser = mi.OptixDenoiser(input_size=img_i.shape[:2])
                        img_denoised = denoiser(img_i)
                        img_denoised = dr.replace_grad(img_denoised, img_i)
                        img_i = img_denoised
                    loss = self.loss_fn(img_i, gt_images[c_i]) / batch_size
                    self.foreach_callback(lambda cb: cb.on_before_backward(self, loss))
                    try:
                        dr.backward(loss)
                        backward_success = True
                    except:
                        camera_params = dataloader.dataset.sample_N(1)
                        camera_params = self.prepare_batch(camera_params)[0]

                        gt_images[c_i] = self._render_gt_tiles(
                            [camera_params], seed + c_i
                        )[0]

                        continue
                    self.foreach_callback(lambda cb: cb.on_after_backward(self))

                    with torch.no_grad():
                        with dr.suspend_grad():
                            total_loss += loss.item()

            for optimizer in self.optimizers:
                self.foreach_callback(
                    lambda cb: cb.on_before_optimizer_step(self, optimizer)
                )
                optimizer.step()

            for scheduler in self.schedulers:
                scheduler.step()

            seed += batch_size

            if heatsplats.is_debug() and (i == 0 or (i + 1) % 100 == 0):
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grads_lists[name].append(param.grad.norm().item())

            with torch.no_grad():
                with dr.suspend_grad():
                    loss_step = total_loss

                    if i == 0 or (i + 1) % 100 == 0:
                        heatsplats.debug(
                            f"Iteration: {i + 1} -> Loss: {loss_step}.",
                        )
                    errors_lists["loss"].append(loss_step)
                    pbar.set_postfix_str(f"Loss: {loss_step:0.4f}")

            dr.flush_malloc_cache()
            torch.cuda.empty_cache()

        self.plot_errors(errors_lists)

        if heatsplats.is_debug():
            self.plot_gradient_norms(grads_lists, log_interval=100, y_log_scale=True)

        return None, None, None

    @abstractmethod
    def _render_gt_tiles(self, batch_cameras, seed) -> list[mi.TensorXf]:
        pass

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
        self, rotating_frames: int = 10, update_scene_bounds: bool = False
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
        renderer = self._get_renderer()

        if update_scene_bounds and self.model.requires_scene_bounds():
            mesh_texless = renderer.mesh_notex_to_mitsuba(self.datamodule.mesh)
            scene_texless = renderer.make_scene(mesh_texless, False)
            scene_min, scene_max = scene_texless.bbox().min, scene_texless.bbox().max
            self.model.set_scene_bounds(scene_min.torch(), scene_max.torch())

        mi_mesh, mi_texture = renderer.mesh_to_mitsuba(self.datamodule.mesh, self.model)

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

    def save_model(self, filename):
        self.model.save_torch(filename)


@heatsplats.register("trainers.uv-texture-mitsuba")
class UvTextureMitsubaTrainer(MitsubaTrainer):
    @dataclass
    class Config(MitsubaTrainer.Config):
        gt_denoise: bool = True
        gt_spp: int = 0

    cfg: Config

    def configure(
        self,
        datamodule: MeshSamplerDataModule,
        **kwargs,
    ):
        super().configure(datamodule, **kwargs)

        self.gt_mesh = self.datamodule.mesh
        if self.datamodule.cfg.merge_tex:
            # Reload the mesh without merging textures to get proper UVs
            self.gt_mesh = load_mesh(
                self.datamodule.cfg.mesh_path,
                show=False,
                merge_tex=False,
                bake_vert_colors=False,
            )

        self.gt_renderer = UVTextureRenderer(self.cfg.renderer)
        mi_mesh = self.gt_renderer.mesh_to_mitsuba(self.gt_mesh)
        self.gt_scene, self.gt_params = self.gt_renderer.make_scene(
            mi_mesh, with_params=True
        )

    def _render_gt_tiles(self, batch_cameras, seed):
        renders = []
        for i, camera_params in enumerate(batch_cameras):
            self.gt_renderer.update_camera_param(self.gt_params, **camera_params)

            img_i = mi.render(
                self.gt_scene,
                spp=self.cfg.gt_spp,
                seed=seed + i,
                seed_grad=seed + i + 1,
            )
            if self.cfg.gt_denoise:
                denoiser = mi.OptixDenoiser(input_size=img_i.shape[:2])
                img_i = denoiser(img_i)
            renders.append(img_i)
        return renders

    def render_gt(self, rotating_frames: int = 10) -> Union[mi.Bitmap, list[mi.Bitmap]]:
        renderer = UVTextureRenderer(self.cfg.renderer)
        mesh = self.gt_mesh

        mi_mesh = renderer.mesh_to_mitsuba(mesh)
        if rotating_frames == 1:
            img = renderer.render(mi_mesh, denoise=True)
            out = mi.Bitmap(img).convert(
                pixel_format=mi.Bitmap.PixelFormat.RGB,
                component_format=mi.Struct.Type.UInt8,
                srgb_gamma=True,
            )
        else:
            out = renderer.rotating_video(mi_mesh, rotating_frames)
        return out

    def render_gt_raw(self):
        renderer = UVTextureRenderer(self.cfg.renderer)
        mi_mesh = renderer.mesh_to_mitsuba(self.gt_mesh)
        img = renderer.render(mi_mesh, denoise=True)
        return img
