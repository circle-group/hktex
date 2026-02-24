from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mitsuba as mi

from trimesh.visual import uv_to_color

import heatsplats
from heatsplats.data import MeshSamplerDataModule
from heatsplats.modules import PointsInfo
from heatsplats.rendering.uv_texture_renderer import UVTextureRenderer
from heatsplats.utils import (
    load_mesh,
    compute_gmap,
    get_grid,
    cart_to_bary_coords,
    interpolate_barycentric_attr,
)
from heatsplats.utils.typing import *

from .base import BaseTrainer


@heatsplats.register("trainers.uv-texture")
class UvTextureTrainer(BaseTrainer):
    @dataclass
    class Config(BaseTrainer.Config):
        pass

    cfg: Config

    def configure(
        self,
        datamodule: MeshSamplerDataModule,
        **kwargs,
    ):
        super().configure(datamodule, **kwargs)

    def prepare_batch(self, data: dict) -> dict:
        data = super().prepare_batch(data)
        face_ids = data["face_id"]
        barys = data["bary"]

        if self.cfg.use_knn_implementation:
            points_info: PointsInfo = self.model.prepare_points(
                mesh=self.mesh,
                eigalbo_interp=self.eigalbo_interp,
                face_ids=face_ids,
                barys=barys,
                pts=None,
            )
            data["weights"] = points_info["weights"]
            data["indices"] = points_info["indices"]
            data["distances"] = points_info["distances"]
        else:
            with torch.profiler.record_function("interpolate_anisotropies"):
                albo_weights = self.eigalbo_interp.interpolate_anisotropies(
                    angles=self.model.angles, scales=self.model.anisotropies
                )

            with torch.profiler.record_function("prepare_points_for_diffusion"):
                points_info: PointsInfo = self.model.prepare_points_for_diffusion(
                    mesh=self.mesh,
                    eigalbo_interp=self.eigalbo_interp,
                    albo_weights=albo_weights,
                    face_ids=face_ids,
                    barys=barys,
                    pts=None,
                )
                data["albo_weights"] = albo_weights

        data["evals"] = points_info["albo_evals"]
        data["pts_iso_evecs"] = points_info["iso_evecs"]
        data["pts_evecs"] = points_info["albo_evecs"]
        data["pts_mass"] = points_info["mass"]
        data["points_info"] = points_info

        return data

    def render_gt(self, rotating_frames: int = 10) -> Union[mi.Bitmap, list[mi.Bitmap]]:
        renderer = UVTextureRenderer(self.cfg.renderer)
        mesh = self.datamodule.mesh

        if self.datamodule.cfg.merge_tex:
            # Reload the mesh without merging textures to get proper UVs
            mesh = load_mesh(
                self.datamodule.cfg.mesh_path,
                show=False,
                merge_tex=False,
                bake_vert_colors=False,
            )

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
        mi_mesh = renderer.mesh_to_mitsuba(self.datamodule.mesh)
        img = renderer.render(mi_mesh, denoise=True)
        return img

    @torch.no_grad()
    def data_dependent_initialisation(self, random_ratio=0.3):
        dataset = self.datamodule.train_dataloader().dataset
        try:
            tex_img_np = np.array(dataset.tex_img)
        except AttributeError:
            return

        gmap = compute_gmap(tex_img_np)

        G = self.model.N_sources
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
        self.model._kernel_locations.data[start_idx:] = cart_pos
        self.model._kernel_face_ids[start_idx:] = valid_face_indices.int()

        # Sample colors from texture ###################################################

        # UVs for the kept kernels (randomly initialized ones)
        kept_face_ids = self.model._kernel_face_ids[:start_idx]
        kept_locations = self.model._kernel_locations[:start_idx]

        kept_vert_ids = self.mesh.get_face_vertices(kept_face_ids)
        kept_barys = self.mesh.cartesian_to_barycentric(kept_locations, kept_vert_ids)

        uvs_kept = interpolate_barycentric_attr(
            self.mesh.faces, kept_face_ids.long(), kept_barys, uvs
        )

        # UVs for the new selected kernels and combine
        uvs_selected = sampled_uvs[valid_indices]
        all_uvs = torch.cat([uvs_kept, uvs_selected], dim=0)
        sampled_colors = (
            uv_to_color(all_uvs.cpu().detach().numpy(), dataset.tex_img)[:, :3] / 255
        )

        sampled_colors = torch.tensor(
            sampled_colors, dtype=torch.float, device=self.device
        )
        mean_colour = torch.mean(sampled_colors, dim=0, keepdim=True)
        residual_colors = sampled_colors - mean_colour

        self.model._mean_colour.copy_(mean_colour)
        self.model._kernel_colours.copy_(self.model._inv_colour_act(residual_colors))
