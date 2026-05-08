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
from heatsplats.data import MeshSamplerDataModule
import heatsplats.rendering as rendering
from heatsplats.rendering.vertex_colours_renderer import VertexColoursRenderer
from heatsplats.rendering.uv_texture_renderer import UVTextureRenderer


import heatsplats.utils as utils
from heatsplats.utils.video import save_video
from heatsplats.utils import ObjectWithCallbacks, load_mesh
from heatsplats.utils.typing import *


class VertexRayTrainer(ObjectWithCallbacks):
    @dataclass
    class Config(ObjectWithCallbacks.Config):
        renderer: dict = field(default_factory=dict)
        renderer_mega_kernel: bool = False

        learning_rate: float = 0.01
        lr_scheduling: bool = True
        steplr_gamma: float = 0.5
        steplr_step_size: int = 15

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

        debug_video_frequency: int = 10

        albedo_init: float = 0.5

        target_size_kb: Optional[float] = None

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

        self._train_mesh = self.datamodule.mesh
        if self.cfg.target_size_kb is not None:
            self._train_mesh = self.datamodule.load_mesh_size_matched(
                self.cfg.target_size_kb, dtype=np.float32
            )
            vc_size_kb = (
                utils.get_vertex_colours_size_bytes(self._train_mesh, dtype=np.float32)
                / 1024.0
            )
            heatsplats.info(
                f"Trying to match to {self.cfg.target_size_kb:.2f} KB, "
                f"final vertex colours size: {vc_size_kb:.2f} KB"
            )

        self.mesh = Mesh.from_trimesh(self._train_mesh, device=self.device)

        self.loss_fn = utils.mitsuba.get_mitsuba_loss(
            self.cfg.loss_type, self.cfg.loss_force_vectorized
        )

        self.opt = mi.ad.Adam(lr=self.cfg.learning_rate, mask_updates=False)

        self._initialise_vertex_colours()
        self.renderer = self._get_renderer()
        self.integrator = self._get_integrator()

    def _initialise_vertex_colours(self):
        n_verts = self.mesh.N_verts
        assert 0 <= self.cfg.albedo_init <= 1
        self.vertex_colours = torch.full(
            (n_verts, 3), self.cfg.albedo_init, dtype=torch.float32
        )

    def _get_renderer(self):
        renderer = VertexColoursRenderer(self.cfg.renderer)
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
        o, d, wavelengths, pos, sensor_idx, hit_face_ids, hit_points, seed_offset = (
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
            "ray_hit_face_ids": hit_face_ids,  # [N_rays]
            "ray_hit_points": hit_points,  # [N_rays, 3]
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

    def optimise(self, n_iter=100, debug_log_dir=None):
        dataloader = self.datamodule.train_dataloader()
        data_iter = iter(dataloader)

        mi_mesh = self.renderer.mesh_to_mitsuba(self._train_mesh, self.vertex_colours)
        scene, params = self.renderer.make_scene(mi_mesh, with_params=True)

        print(params)

        self.opt["mesh.vertex_color"] = params["mesh.vertex_color"]
        params.update(self.opt)

        seed = 0
        grad_spp = self.cfg.grad_spp
        global_step = 0

        errors_lists = self._build_errors_lists()

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
            ray_o, ray_d, ray_wavelengths = batches["ray_data"]
            ray_pos: Tensor = batches["ray_pos"]
            ray_sidx: Tensor = batches["ray_sensor_idx"]
            ray_hit_points: Tensor = batches["ray_hit_points"]
            ray_hit_fids: Tensor = batches["ray_hit_face_ids"]
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
                ray, batch_pos, batch_sidx = self._make_rays(
                    ray_o, ray_d, ray_idx, grad_spp, ray_wavelengths, ray_pos, ray_sidx
                )
                batch_hit_pts = ray_hit_points[ray_idx].to(self.device)  # [B,3]
                batch_hit_fids = ray_hit_fids[ray_idx].to(self.device)  # [B]

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
                    params=params,
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

                self.opt.step()
                self.opt["mesh.vertex_color"] = dr.clamp(
                    self.opt["mesh.vertex_color"], 0.0, 1.0
                )
                params.update(self.opt)

                self.vertex_colours = params["mesh.vertex_color"].torch()

                with torch.no_grad():
                    with dr.suspend_grad():
                        loss_val = loss.item()
                        total_loss += loss_val
                        pbar_inner.set_postfix_str(f"Loss: {loss_val:0.4f}")

                global_step += 1

                dr.flush_malloc_cache()
                dr.flush_malloc_cache()
                dr.flush_malloc_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
            # # TODO: maybe seperate per epoch and per step schedulers
            # for scheduler in self.schedulers:
            #     scheduler.step()
            if self.cfg.lr_scheduling:
                gamma, step_size = self.cfg.steplr_gamma, self.cfg.steplr_step_size
                if (epoch + 1) % step_size == 0:
                    prev_lr = self.opt.learning_rate()
                    lr = prev_lr * gamma
                    heatsplats.info(f"Updating learning rate from {prev_lr} to {lr}")
                    self.opt.set_learning_rate(lr)

            with torch.no_grad():
                with dr.suspend_grad():
                    n_steps = int(ray_batches.shape[0])
                    loss_epoch = total_loss / max(n_steps, 1)

                    errors_lists["loss"].append(loss_epoch)
                    pbar.set_postfix_str(f"Loss: {loss_epoch:0.4f}")

                    if epoch == 0 or (epoch + 1) % 100 == 0:
                        heatsplats.debug(
                            f"Iteration: {epoch + 1} -> Loss: {loss_epoch}."
                        )

            dr.flush_malloc_cache()
            torch.cuda.empty_cache()

        self.vertex_colours = params["mesh.vertex_color"].torch()

        return None, None, None

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
        Render the mesh with the resulting heat kernel texture. This is always rendered
        with the heat kernel texture, so it is not defined in subclasses.
        Args:
            rotating_frames (int): Number of frames for rotation.
                If 1, render a single image.
        Returns:
            mi.Bitmap or list[mi.Bitmap]: The rendered image(s).
        """
        renderer = self._get_renderer()

        mi_mesh = renderer.mesh_to_mitsuba(self._train_mesh, self.vertex_colours)

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

    def _build_errors_lists(self) -> dict[str, list[float]]:
        return {"loss": []}

    def save_model(self, filename):
        model_torch = {"vertex_colours": self.vertex_colours}
        torch.save(model_torch, filename)
        torch_size = os.path.getsize(filename)

        model_npz = {k: v.detach().cpu().numpy() for k, v in model_torch.items()}
        npz_filename = filename.replace(".pt", ".npz")
        np.savez_compressed(npz_filename, **model_npz)
        npz_size = os.path.getsize(npz_filename)

        return torch_size / 1024, npz_size / 1024


@heatsplats.register("trainers.uv-texture-vertex-ray")
class UvTextureVertexRayTrainer(VertexRayTrainer):
    @dataclass
    class Config(VertexRayTrainer.Config):
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
