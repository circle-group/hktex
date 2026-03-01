from dataclasses import dataclass, field
from abc import abstractmethod
import numpy as np
from termcolor import colored
from functools import partial
import trimesh
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import mitsuba as mi
import drjit as dr

from trimesh.visual import uv_to_color

import heatsplats
from heatsplats.modules import (
    Mesh,
)
from heatsplats.modules.base import TextureModel
from heatsplats.modules.heat_kernel_model_knn import HeatKernelModelKNN
from heatsplats.data import MeshSamplerDataModule
import heatsplats.rendering as rendering
from heatsplats.rendering.torch_texture_renderer import TorchTextureRenderer
from heatsplats.rendering.uv_texture_renderer import UVTextureRenderer
from heatsplats.modules import (
    HeatKernelModel,
    HeatKernelModelKNN,
    HeatKernelTexture,
    HeatKernelTextureKNN,
)


import heatsplats.utils as utils
from heatsplats.utils.video import save_video
from heatsplats.utils import ObjectWithCallbacks, load_mesh
from heatsplats.utils import (
    compute_gmap,
    get_grid,
    cart_to_bary_coords,
    interpolate_barycentric_attr,
)
from heatsplats.utils.typing import *

from .utils import parse_optimizers_and_schedulers


class MitsubaRayTrainer(ObjectWithCallbacks):
    @dataclass
    class Config(ObjectWithCallbacks.Config):
        network_type: str = "modules.mlp-texture-network"
        network: dict = field(default_factory=dict)

        optimizers: list = field(default_factory=list)

        renderer_type: str = "renderer.torch-texture"
        renderer: dict = field(default_factory=dict)
        renderer_mega_kernel: bool = False

        # denoise_ad_prop: bool = False
        spp: int = 0
        grad_spp: int = 32

        loss_type: str = "l1_loss"  # "mse_loss"
        loss_force_vectorized: bool = False

        use_linear_gamma_space: bool = True
        use_hdr: bool = False

        batch_size: int = 128
        shuffle_rays: bool = True
        ray_sample_rate: float = 1.0

        ray_integrator: str = "rb_ray"
        base_integrator: str = "prb"
        integrator_max_depth: int = 5
        integrator_hide_emitters: bool = False

        data_initialisation_random_ratio: float = 0.3

        use_knn_implementation: bool = False

        debug_video_frequency: int = 10

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
        self.model: TextureModel = heatsplats.find(self.cfg.network_type)(
            self.cfg.network, mesh=self.mesh
        )
        self.loss_fn = utils.mitsuba.get_mitsuba_loss(
            self.cfg.loss_type, self.cfg.loss_force_vectorized
        )

        self.optimizers, self.schedulers = parse_optimizers_and_schedulers(
            self.cfg.optimizers, self.model
        )

        self.renderer = self._get_renderer()
        self.integrator = self._get_integrator()

    def _get_renderer(self):
        renderer: TorchTextureRenderer = heatsplats.find(self.cfg.renderer_type)(
            self.cfg.renderer
        )
        renderer.mega_kernel(
            self.cfg.renderer_mega_kernel, no_loops=True, no_opt_calls=True
        )
        return renderer

    def _get_integrator(self):
        integrator_dict = {
            "type": self.cfg.ray_integrator,
            "integrator": {
                "type": self.cfg.base_integrator,
                "max_depth": self.cfg.integrator_max_depth,
                "hide_emitters": self.cfg.integrator_hide_emitters,
            },
        }
        integrator: rendering.ray_integrator.RBRayIntegrator = mi.load_dict(
            integrator_dict
        )
        return integrator

    def prepare_batch(self, data: dict, scene: mi.Scene, seed: int) -> dict:
        batch_cameras = [dict() for _ in range(data["batch_size"])]
        for k, v in data["cameras"].items():
            for i, c_v in enumerate(v):
                batch_cameras[i][k] = c_v.item()
        sensors = []
        camera_overrides = data["extra_overrides"]
        for camera in batch_cameras:
            sensor = self.renderer.get_camera_params(**camera, **camera_overrides)
            sensors.append(mi.load_dict(sensor))

        # Sample rays
        # We keep the origins (o), directions(d), and wavelengths associated with each ray
        o, d, wavelengths, pos, sensor_idx, seed_offset = (
            rendering.sample_intersecting_rays_multiple_sensors(
                self.integrator, scene, sensors, seed=seed
            )
        )

        batch_size, shuffle_rays = self.cfg.batch_size, self.cfg.shuffle_rays
        N_rays = o.shape[0]
        # Shuffle indices of rays
        indices = torch.randperm(N_rays) if shuffle_rays else torch.arange(N_rays)
        if self.cfg.ray_sample_rate < 1.0:
            assert 0.0 < self.cfg.ray_sample_rate <= 1.0
            N_rays = int(self.cfg.ray_sample_rate * N_rays)
            indices = indices[:N_rays]
        max_its = N_rays // batch_size
        if max_its == 0:
            raise RuntimeError(f"number of rays less than batch size, check config")
        batches = indices[: max_its * batch_size].reshape(-1, batch_size)

        batch = {
            "ray_data": (o, d, wavelengths),  # ray origin, direction, wavelengths
            "ray_pos": pos,  # Screen space positions
            "ray_sensor_idx": sensor_idx,  # Sensor idx for the ray
            "batches": batches,  # Ray batch indices
            "sensors": sensors,  # Sensors used to sample rays
            "batch_cameras": batch_cameras,  # Cameras used to sample rays
            "seed_offset": seed_offset,  # How much to offset ray sampling seed
        }
        return batch

    def _move_to_device(self, obj):
        if isinstance(obj, Tensor):
            return obj.to(self.device)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        else:
            return obj

    def _make_rays(
        self,
        ray_o: Tensor,
        ray_d: Tensor,
        ray_idx: Tensor,
        grad_spp: int,
        ray_wavelengths: Tensor,
        ray_pos: Tensor,
        ray_sidx: Tensor,
    ):
        batch_o = ray_o[ray_idx].to(self.device).repeat_interleave(grad_spp, dim=0)
        batch_d = ray_d[ray_idx].to(self.device).repeat_interleave(grad_spp, dim=0)
        ray = mi.RayDifferential3f(
            o=mi.Point3f(batch_o.t()),
            d=mi.Vector3f(batch_d.t()),
            wavelengths=ray_wavelengths,
        )
        batch_pos = ray_pos[ray_idx].to(self.device)
        batch_sidx = ray_sidx[ray_idx].to(self.device)
        return ray, batch_pos, batch_sidx

    @abstractmethod
    def data_dependent_initialisation(self, **kwargs):
        raise NotImplementedError

    def optimise(self, n_iter=100, debug_log_dir=None):
        dataloader = self.datamodule.train_dataloader()
        data_iter = iter(dataloader)

        random_ratio = self.cfg.data_initialisation_random_ratio
        if random_ratio < 1.0:
            self.data_dependent_initialisation(random_ratio=random_ratio)

        heatsplats.debug(f"INITIAL -> {self._debug_opt_params_string()}")

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
        grad_spp = self.cfg.grad_spp

        errors_lists = self._build_errors_lists()
        grads_lists = {k: [] for k, _ in self.model.named_parameters()}

        for epoch in (pbar := tqdm(range(n_iter), disable=True)):
            if debug_log_dir is not None and (
                epoch == 0 or epoch % self.cfg.debug_video_frequency == 0
            ):
                current_rnd = self.render_result(self.cfg.renderer.n_rotating_frames)
                save_video(
                    current_rnd, os.path.join(debug_log_dir, f"iter_{epoch}.mp4")
                )

            data = next(data_iter)
            batches = self.prepare_batch(data, scene, seed)
            ray_o: Tensor
            ray_d: Tensor
            (ray_o, ray_d, ray_wavelengths) = batches["ray_data"]
            ray_pos: Tensor = batches["ray_pos"]
            ray_sidx: Tensor = batches["ray_sensor_idx"]
            ray_batches = batches["batches"]
            seed_offset = batches["seed_offset"]
            seed += seed_offset
            # print("A")
            total_loss = 0.0
            for it, ray_idx in (
                pbar_inner := tqdm(
                    enumerate(ray_batches),
                    total=ray_batches.shape[0],
                    desc=f"Epoch {epoch+1}",
                    # leave=False,
                )
            ):
                # print("B")
                for optimizer in self.optimizers:
                    self.foreach_callback(
                        lambda cb: cb.on_before_zero_grad(self, optimizer)
                    )
                    optimizer.zero_grad()

                ray, batch_pos, batch_sidx = self._make_rays(
                    ray_o, ray_d, ray_idx, grad_spp, ray_wavelengths, ray_pos, ray_sidx
                )

                with torch.no_grad():
                    with dr.suspend_grad():
                        seed += 1
                        target = self._render_gt_rays(ray, batch_pos, batch_sidx, seed)

                # if self.cfg.use_knn_implementation:
                #     self.prepare_knn()

                # Render target image
                seed += 1
                L = rendering.render_ray(
                    scene,
                    params=grad_params,
                    integrator=self.integrator,
                    ray=ray,
                    spp=grad_spp,
                    seed=seed,
                )

                # Integrate samples per pixel
                L_integrated = rendering.integrate_ray_samples(L, grad_spp)

                if not self.cfg.use_hdr:
                    L_integrated = dr.clip(L_integrated, 0.0, 1.0)
                    target = dr.clip(target, 0.0, 1.0)

                if self.cfg.use_linear_gamma_space:
                    L_integrated = utils.linear_to_gamma_dr(
                        utils.to_log_dr(L_integrated * (1 << 16))
                    )
                    target = utils.linear_to_gamma_dr(
                        utils.to_log_dr(target * (1 << 16))
                    )
                # loss = self.loss_fn(L_integrated, target)
                err = self.loss_fn(L_integrated, target, reduction="none")
                per_ray_loss = dr.dot(err, mi.Color3f(1.0))
                loss = dr.mean(per_ray_loss, axis=None)

                self.foreach_callback(lambda cb: cb.on_before_backward(self, loss))
                dr.backward(loss)
                self.foreach_callback(lambda cb: cb.on_after_backward(self))

                for optimizer in self.optimizers:
                    self.foreach_callback(
                        lambda cb: cb.on_before_optimizer_step(self, optimizer)
                    )
                    optimizer.step()

                self._post_optimizer_step()

                with torch.no_grad():
                    with dr.suspend_grad():
                        loss_val = loss.item()
                        total_loss += loss_val
                        pbar_inner.set_postfix_str(f"Loss: {loss_val:0.4f}")

                if self.cfg.use_knn_implementation:
                    self.reset_knn()

                dr.flush_malloc_cache()
                dr.flush_malloc_cache()
                dr.flush_malloc_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
            # TODO: maybe seperate per epoch and per step schedulers
            for scheduler in self.schedulers:
                scheduler.step()

            if heatsplats.is_debug():  # and (epoch == 0 or (epoch + 1) % 10 == 0):
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grads_lists[name].append(param.grad.norm().item())

            with torch.no_grad():
                with dr.suspend_grad():
                    n_steps = int(ray_batches.shape[0])
                    loss_epoch = total_loss / max(n_steps, 1)
                    errors = self._current_error_snapshot()

                    for k in errors_lists.keys():
                        if k == "loss":
                            continue
                        if k in errors and torch.is_tensor(errors[k]):
                            errors_lists[k].append(errors[k].item())

                    errors_lists["loss"].append(loss_epoch)
                    pbar.set_postfix_str(f"Loss: {loss_epoch:0.4f}")

                    if epoch == 0 or (epoch + 1) % 100 == 0:
                        heatsplats.debug(
                            f"Iteration: {epoch + 1} -> Loss: {loss_epoch}. {errors['printables']}"
                        )

            dr.flush_malloc_cache()
            torch.cuda.empty_cache()

        heatsplats.debug(f"FINAL -> {self._debug_opt_params_string()}")

        self.plot_errors(errors_lists)

        if heatsplats.is_debug():
            self.plot_gradient_norms(grads_lists, log_interval=100, y_log_scale=True)

        return None, None, None

    # def prepare_knn(self):
    #     self.model: HeatKernelModelKNN
    #     return self.model.prepare_kernels()

    def reset_knn(self):
        self.model: HeatKernelModelKNN
        return self.model.reset()

    def _post_optimizer_step(self):
        if hasattr(self.model, "post_optimizer_step"):
            self.model.post_optimizer_step()

    @abstractmethod
    def _render_gt_rays(self, ray, pos, s_idx, seed) -> mi.TensorXf:
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

        # if self.cfg.use_knn_implementation:
        #     self.prepare_knn()
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

    def _get_hk_texture_or_none(
        self,
    ) -> Optional[Union[HeatKernelTexture, HeatKernelTextureKNN]]:
        model = self.model

        if isinstance(model, (HeatKernelModel, HeatKernelModelKNN)):
            inner = getattr(model, "model", None)
            if isinstance(inner, (HeatKernelTexture, HeatKernelTextureKNN)):
                return inner

        return None

    def _build_errors_lists(self) -> dict[str, list[float]]:
        texture = self._get_hk_texture_or_none()
        if texture is not None and hasattr(texture, "splat_param_keys"):
            out = {k: [] for k in texture.splat_param_keys}
            out["loss"] = []
            return out
        return {"loss": []}

    def _current_error_snapshot(self) -> dict[str, Any]:
        texture = self._get_hk_texture_or_none()
        if texture is None:
            return {"printables": ""}

        snapshot = {
            "kernel_colours": texture.kernel_colours.detach().mean(),
            "angles": texture.angles.detach().mean(),
            "anisotropies": texture.anisotropies.detach().mean(),
            "sharpnesses": texture.sharpnesses.detach().mean(),
            "thresholds": texture.thresholds.detach().mean(),
        }
        snapshot["printables"] = (
            f"angles={snapshot['angles'].item():.4f}, "
            f"anis={snapshot['anisotropies'].item():.4f}, "
            f"sharp={snapshot['sharpnesses'].item():.4f}, "
            f"thresh={snapshot['thresholds'].item():.4f}, "
            f"col={snapshot['kernel_colours'].item():.4f}"
        )
        return snapshot

    def _debug_opt_params_string(self) -> str:
        texture = self._get_hk_texture_or_none()
        if texture is not None and hasattr(texture, "colored_print_opt_params"):
            return texture.colored_print_opt_params

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"trainable_params={n_trainable}"

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
            if not norm_list:
                heatsplats.warn(f"Gradient list for {name} is empty. Skipping.")
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

    def save_model(self, filename):
        self.model.save_torch(filename)
        torch_size = os.path.getsize(filename)

        npz_filename = filename.replace(".pt", ".npz")
        self.model.save_numpy_npz(npz_filename)
        npz_size = os.path.getsize(npz_filename)
        return torch_size / 1024, npz_size / 1024


@heatsplats.register("trainers.uv-texture-mitsuba-ray")
class UvTextureMitsubaRayTrainer(MitsubaRayTrainer):
    @dataclass
    class Config(MitsubaRayTrainer.Config):
        gt_spp: int = 0  # Ignored during training until we implement the todo below

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
        self._gt_tex_img = self._get_gt_texture_image()

        self.gt_renderer = UVTextureRenderer(self.cfg.renderer)
        mi_mesh = self.gt_renderer.mesh_to_mitsuba(self.gt_mesh)
        self.gt_scene, self.gt_params = self.gt_renderer.make_scene(
            mi_mesh, with_params=True
        )

    # TODO: This is temporary and mirrors training
    #       We can instead use a better integrator (path with -1 max_depth + higher spp)
    #       and then sample using the screen space pos + sensor idx
    #       This could also be faster training/lower memory

    def _render_gt_rays(self, ray, pos, s_idx, seed) -> mi.TensorXf:
        # Render target image
        L = rendering.render_ray(
            self.gt_scene,
            integrator=self.integrator,
            ray=ray,
            spp=self.cfg.grad_spp,
            seed=seed,
        )

        # Integrate samples per pixel
        L_integrated = rendering.integrate_ray_samples(L, self.cfg.grad_spp)
        return L_integrated

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

    @torch.no_grad()
    def data_dependent_initialisation(self, random_ratio=0.3):
        tex_img = self._gt_tex_img
        tex_img_np = np.array(tex_img)

        gmap = compute_gmap(tex_img_np)

        G = self.model.model.N_sources
        num_random = round(random_ratio * G)
        num_selected = G - num_random

        if num_selected <= 0:
            return

        pixel_xy = (
            get_grid(h=tex_img_np.shape[0], w=tex_img_np.shape[1])
            .to(device=self.device)
            .reshape(-1, 2)
        )

        # Oversample to account for invalid UVs (e.g. padding/background)
        oversample_factor = 10
        num_to_sample = min(gmap.shape[0], num_selected * oversample_factor)

        selected_candidates = np.random.choice(
            gmap.shape[0], num_to_sample, replace=False, p=gmap
        )
        sampled_uvs = pixel_xy[selected_candidates]

        uvs = self.mesh.uv.to(self.device)
        face_uvs = uvs[self.mesh.faces]  # (F, 3, 2)

        # Reshape face_uvs to (1, 3, F, 2) so that cart_to_bary_coords slices the vertex dim correctly
        # cart_to_bary_coords expects verts[:, 0] to be the first vertex of the triangle
        face_uvs_reshaped = face_uvs.permute(1, 0, 2).unsqueeze(0)

        # Chunked validity check to avoid OOM
        chunk_size = 1024
        valid_indices_list = []
        match_face_indices_list = []
        num_found = 0

        for i in range(0, num_to_sample, chunk_size):
            if num_found >= num_selected:
                break

            chunk_uvs = sampled_uvs[i : i + chunk_size]
            barys_chunk = cart_to_bary_coords(chunk_uvs.unsqueeze(1), face_uvs_reshaped)
            u, v, w = barys_chunk[..., 0], barys_chunk[..., 1], barys_chunk[..., 2]

            mask = (u >= -1e-4) & (v >= -1e-4) & (w >= -1e-4)
            has_match, match_idx = mask.max(dim=1)

            valid_in_chunk = torch.where(has_match)[0]

            if len(valid_in_chunk) > 0:
                valid_indices_list.append(valid_in_chunk + i)
                match_face_indices_list.append(match_idx[valid_in_chunk])
                num_found += len(valid_in_chunk)

        if num_found == 0:
            return

        valid_indices = torch.cat(valid_indices_list)
        valid_face_indices = torch.cat(match_face_indices_list)

        # Truncate to desired number
        if len(valid_indices) > num_selected:
            valid_indices = valid_indices[:num_selected]
            valid_face_indices = valid_face_indices[:num_selected]

        # Recompute barys for the selected valid points (1-to-1)
        final_uvs = sampled_uvs[valid_indices]
        final_face_uvs = face_uvs[valid_face_indices]
        barys = cart_to_bary_coords(final_uvs, final_face_uvs)
        barys = torch.clamp(barys, 0.0, 1.0)
        barys = barys / (barys.sum(dim=-1, keepdim=True) + 1e-8)

        mesh_faces = self.mesh.faces.to(self.device)
        mesh_verts = self.mesh.verts.to(self.device)

        chosen_faces = mesh_faces[valid_face_indices.long()]
        chosen_verts = mesh_verts[chosen_faces.long()]

        cart_pos = (chosen_verts * barys.unsqueeze(-1)).sum(dim=1)

        # Update kernel positions in the model
        start_idx = G - len(valid_indices)
        self.model.model._kernel_locations.data[start_idx:] = cart_pos
        self.model.model._kernel_face_ids[start_idx:] = valid_face_indices.int()

        # Sample colors from texture ###################################################

        # UVs for the kept kernels (randomly initialized ones)
        kept_face_ids = self.model.model._kernel_face_ids[:start_idx]
        kept_locations = self.model.model._kernel_locations[:start_idx]

        kept_vert_ids = self.mesh.get_face_vertices(kept_face_ids)
        kept_barys = self.mesh.cartesian_to_barycentric(kept_locations, kept_vert_ids)

        uvs_kept = interpolate_barycentric_attr(
            self.mesh.faces, kept_face_ids.long(), kept_barys, uvs
        )

        # UVs for the new selected kernels and combine
        uvs_selected = sampled_uvs[valid_indices]
        all_uvs = torch.cat([uvs_kept, uvs_selected], dim=0)
        sampled_colors = (
            uv_to_color(all_uvs.cpu().detach().numpy(), tex_img)[:, :3] / 255
        )

        sampled_colors = torch.tensor(
            sampled_colors, dtype=torch.float, device=self.device
        )
        mean_colour = torch.mean(sampled_colors, dim=0, keepdim=True)
        residual_colors = sampled_colors - mean_colour

        self.model.model._mean_colour.copy_(mean_colour)
        self.model.model._kernel_colours.copy_(
            self.model.model._inv_colour_act(residual_colors)
        )

    def _get_gt_texture_image(self):
        try:
            tex_img = self.gt_mesh.visual.material.baseColorTexture
            if tex_img is None:
                raise AttributeError
        except AttributeError:
            tex_img = self.gt_mesh.visual.material.image
            if tex_img is None:
                raise AttributeError
        return tex_img
