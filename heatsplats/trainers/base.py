from dataclasses import dataclass, field
from abc import abstractmethod
import numpy as np
from termcolor import colored
import trimesh
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import heatsplats
from heatsplats.modules import (
    Mesh,
    Model,
    GeodesicTracer,
    GeodesicOpt,
    EigenAlboInterpolation,
)
from heatsplats.data import MeshSamplerDataModule

import heatsplats.utils as utils
from heatsplats.utils import BaseObject
from heatsplats.utils.typing import *

from .utils import parse_optimizers


class BaseTrainer(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        tracer_type: str = ""
        tracer: dict = field(default_factory=dict)

        eigen_albo: dict = field(default_factory=dict)
        model: dict = field(default_factory=dict)

        optimizers: list = field(default_factory=list)

    cfg: Config

    def configure(
        self,
        datamodule: MeshSamplerDataModule,
        **kwargs,
    ):
        super().configure()

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

    def prepare_batch(self, data: dict) -> dict:
        for k, v in data.items():
            if isinstance(v, Tensor):
                data[k] = v.to(self.device)
        return data

    def optimise(self, n_iter=100):
        dataloader = self.datamodule.train_dataloader()
        data_iter = iter(dataloader)

        B = self.model.N_sources

        heatsplats.debug(f"INITIAL -> {self.model.colored_print_opt_params}")

        errors_lists = {k: [] for k in self.model.splat_param_keys}
        errors_lists["loss"] = []

        for i in (pbar := tqdm(range(n_iter))):
            data = next(data_iter)
            data = self.prepare_batch(data)

            pos: Tensor = data["pos"]
            gt_colours: Tensor = data["colour"]
            evals: Tensor = data["evals"]
            pts_evecs: Tensor = data["pts_evecs"]
            pts_mass: Tensor = data["pts_mass"]
            albo_weights = data["albo_weights"]

            P = pos.shape[0]

            colours: Float[Tensor, "B P L"] = torch.zeros(
                [B, P, self.model.kernel_dim],
                device=self.device,
            )

            kernel_vert_idx = self.mesh.get_face_vertices(self.model.kernel_face_ids)
            barycentric_coords = self.mesh.cartesian_to_barycentric(
                self.model.kernel_locations, kernel_vert_idx
            )
            self.model.save_barycentric_locations(barycentric_coords)

            kernel_evecs, kernel_mass = self.eigalbo_interp.barycentric_albo_gaussians(
                albo_weights=albo_weights,
                barycentric_coords=barycentric_coords,
                vert_idx=kernel_vert_idx,
            )

            colours: Float[Tensor, "B P+1 L"] = torch.cat(
                (colours, self.model.kernel_colours.unsqueeze(1)), dim=1
            )
            pts_evecs: Float[Tensor, "B P+1 K"] = torch.cat(
                ((pts_evecs, kernel_evecs.unsqueeze(1))), dim=1
            )
            pts_mass: Float[Tensor, "B P+1"] = torch.cat(
                (pts_mass.expand(B, -1), kernel_mass.unsqueeze(-1)), dim=1
            )

            colours = utils.heat_diffusion_reduce(
                colours,
                pts_mass,
                evals,
                pts_evecs,
                self.model.diff_times,
            )
            colours = colours[:P]

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

            with torch.no_grad():
                errors = self._errors
                loss_step = loss.item()
                if i == 0 or (i + 1) % 100 == 0:
                    heatsplats.info(
                        f"Iteration: {i + 1} -> Loss: {loss_step}. {errors['printables']}",
                    )

                for k in self.model.splat_param_keys:
                    if k in errors:
                        errors_lists[k].append(errors[k].item())
                errors_lists["loss"].append(loss_step)
                pbar.set_postfix_str(f"Loss: {loss_step:0.4f}")

        heatsplats.debug(f"FINAL -> {self.model.colored_print_opt_params}")

        self.plot_errors(errors_lists)
        v_colours = self.compute_vertex_colours()
        return v_colours, gt_colours, init_colours

    def compute_vertex_colours(self):
        B, V = self.model.N_sources, self.mesh.N_verts
        v_colours: Float[Tensor, "B V L"] = torch.zeros(
            [B, V, self.model.kernel_dim],
            device=self.device,
        )

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

        v_colours: Float[Tensor, "B V+1 L"] = torch.cat(
            (v_colours, self.model.kernel_colours.unsqueeze(1)), dim=1
        )
        albo_evecs: Float[Tensor, "B V+1 K"] = torch.cat(
            ((albo_evecs, kernel_evecs.unsqueeze(1))), dim=1
        )
        mass: Float[Tensor, "B V+1"] = torch.cat(
            (mass.expand(B, -1), kernel_mass.unsqueeze(-1)), dim=1
        )

        v_colours = utils.heat_diffusion_reduce(
            v_colours,
            mass,
            albo_evals,
            albo_evecs,
            self.model.diff_times,
        )
        v_colours = v_colours[:V]
        v_colours = self.model(v_colours)
        return v_colours

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
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.plot(errors_lists["loss"], label="Loss")
        ax.set_title("Loss per step")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")

        ax.legend()

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
