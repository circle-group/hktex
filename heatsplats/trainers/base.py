from dataclasses import dataclass, field
from abc import abstractmethod
import numpy as np
from termcolor import colored
import trimesh

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import heatsplats
from heatsplats.modules import Mesh, GeodesicTracer, GeodesicOpt, EigenAlboInterpolation
from heatsplats.data import MeshSamplerDataModule

import heatsplats.utils as utils
from heatsplats.utils import BaseObject
from heatsplats.utils.typing import *


@dataclass
class LearningRateConfig:
    multiplier: float = 1.0
    centres: float = 1e-1
    colors: float = 1e-3
    anisotropies: float = 1e-3
    angles: float = 1e-3
    diff_times: float = 1e-3
    out_net: Union[None, float] = 1e-3


class BaseTrainer(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        tracer_type: str = ""
        tracer: dict = field(default_factory=dict)

        eigen_albo: dict = field(default_factory=dict)

        n_sources: int = 128
        kernel_dim: int = 32

        normalize_colours: bool = False

        lrs: LearningRateConfig = field(default_factory=LearningRateConfig)

    cfg: Config

    def configure(
        self,
        datamodule: MeshSamplerDataModule,
        **kwargs,
    ):
        super().configure()

        self.n_sources = self.cfg.n_sources
        self.kernel_dim = self.cfg.kernel_dim
        self.normalize_colours = self.cfg.normalize_colours
        self._lrs = self.cfg.lrs
        self._lr_mult = self._lrs.multiplier

        self.datamodule = datamodule

        self.mesh = Mesh.from_trimesh(self.datamodule.mesh, device=self.device)

        self.eigalbo_interp = EigenAlboInterpolation(self.cfg.eigen_albo, self.mesh)
        self.tracer: GeodesicTracer = heatsplats.find(self.cfg.tracer_type)(
            self.cfg.tracer, self.mesh
        )

        self._diff_time_scaler_func = lambda x: 10 ** (4 * torch.tanh(x) - 2)
        self._angle_scaler = torch.pi
        self._angle_act = lambda x: self._angle_scaler * F.hardsigmoid(x)
        self._anis_act = lambda x: torch.exp(x)
        self._colour_act = lambda x: x

        if self._lrs.out_net is not None and self._lrs.out_net > 0:
            self.out_net = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.kernel_dim, 2 * self.kernel_dim),
                nn.ReLU(),
                nn.Linear(2 * self.kernel_dim, 3),
                nn.Sigmoid(),
            ).to(self.device)
        else:
            self.out_net = None

        self._splats, self._optims = self._make_splats_and_optimisers(self.n_sources)

    def _make_splats_and_optimisers(self, n_sources: int, **kwargs):
        kernel_colours = torch.randn(
            (n_sources, self.kernel_dim), dtype=torch.float, device=self.device
        )
        angles = torch.randn(n_sources, device=self.device)
        anisotropies = torch.randn(n_sources, device=self.device)
        diff_times = torch.rand(n_sources, device=self.device)

        # Sample face and barycentric location on face
        kernel_face_ids = torch.randint(
            0, self.mesh.N_faces, (n_sources,), device=self.device
        )
        kernel_locations = utils.uniform_sample_triangle(
            torch.rand((n_sources, 2), device=self.device)
        )
        # Convert to cartesian coordinates
        kernel_vert_idx = self.mesh.get_face_vertices(kernel_face_ids)
        kernel_locations = self.mesh.barycentric_to_cartesian(
            kernel_locations, kernel_vert_idx
        )

        params = [
            # name, value, lr
            ("kernel_colours", torch.nn.Parameter(kernel_colours), self._lrs.colors),
            ("angles", torch.nn.Parameter(angles), self._lrs.angles),
            ("anisotropies", torch.nn.Parameter(anisotropies), self._lrs.anisotropies),
            ("diff_times", torch.nn.Parameter(diff_times), self._lrs.diff_times),
        ]
        self._splat_param_keys = [x[0] for x in params]
        for name, (param, lr) in kwargs.items():
            params.append((name, param, lr))

        if self.out_net is not None:
            params.append(("out_net", self.out_net.parameters(), self._lrs.out_net))

        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(self.device)

        optimisers = {
            "splats": torch.optim.Adam(
                [
                    {"params": splats[name], "lr": self._lr_mult * lr}
                    for name, _, lr in params
                ],
            )
        }

        self._splat_param_keys.append("kernel_locations")
        splats["kernel_locations"] = nn.Parameter(kernel_locations)
        self._kernel_face_ids = kernel_face_ids
        optimisers["kernel_locations"] = GeodesicOpt(
            [
                {
                    "params": [splats["kernel_locations"]],
                    "lr": self._lr_mult * self._lrs.centres,
                    "face_ids": [self._kernel_face_ids],
                }
            ],
            tracer=self.tracer,
        )

        return splats, optimisers

    @property
    def kernel_colours(self) -> torch.Tensor:
        return self._colour_act(self._splats["kernel_colours"])

    @property
    def angles(self) -> torch.Tensor:
        return self._angle_act(self._splats["angles"])

    @property
    def anisotropies(self) -> torch.Tensor:
        return self._anis_act(self._splats["anisotropies"])

    @property
    def diff_times(self) -> torch.Tensor:
        return self._diff_time_scaler_func(self._splats["diff_times"])

    @property
    def kernel_locations(self) -> torch.Tensor:
        return self._splats["kernel_locations"]

    @property
    def kernel_face_ids(self) -> torch.Tensor:
        return self._kernel_face_ids

    def prepare_batch(self, data: dict) -> dict:
        for k, v in data.items():
            if isinstance(v, Tensor):
                data[k] = v.to(self.device)
        return data

    def optimise(self, n_iter=100):
        dataloader = self.datamodule.train_dataloader()
        data_iter = iter(dataloader)

        B, V = self.n_sources, self.mesh.N_verts

        heatsplats.debug(f"INITIAL -> {self._colored_print_opt_params}")

        errors_lists = {k: [] for k in self._splats.keys()}

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
                [B, P, self.kernel_dim],
                device=self.device,
            )

            kernel_vert_idx = self.mesh.get_face_vertices(self.kernel_face_ids)
            barycentric_coords = self.mesh.cartesian_to_barycentric(
                self.kernel_locations, kernel_vert_idx
            )
            # TODO: Can be cleaned a bit, saved for tracer in optimizer
            setattr(
                self._splats["kernel_locations"],
                "bary_coords",
                barycentric_coords.detach(),
            )

            kernel_evecs, kernel_mass = (
                self.eigalbo_interp.barycentric_albo_batchwise_next(
                    albo_weights=albo_weights,
                    barycentric_coords=barycentric_coords,
                    vert_idx=kernel_vert_idx,
                )
            )

            colours: Float[Tensor, "B P+1 L"] = torch.cat(
                (colours, self.kernel_colours.unsqueeze(1)), dim=1
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
                self.diff_times,
            )
            colours = colours[:P]

            # Postprocess
            if self.out_net is not None:
                colours = self.out_net(colours)
            if self.normalize_colours:
                colours = utils.normalise_colours(colours)

            if i == 0:
                init_colours = colours.clone().detach()

            # Compute loss and backpropagate
            loss = F.mse_loss(colours, gt_colours, reduction="sum") / colours.shape[0]

            loss.backward()

            with torch.no_grad():
                if i == 0 or (i + 1) % 100 == 0:
                    for name in self._splat_param_keys:
                        heatsplats.debug(
                            f"{name}: {self._splats[name].grad.data.norm(2)}"
                        )

            for optimizer in self._optims.values():
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                errors = self._errors
                if i == 0 or (i + 1) % 100 == 0:
                    heatsplats.info(
                        f"Iteration: {i + 1} -> Loss: {loss.item()}. {errors['printables']}",
                    )

                for k in self._splat_param_keys:
                    if k in errors:
                        errors_lists[k].append(errors[k].item())
                pbar.set_postfix_str(f"Loss: {loss.item():0.4f}")

        heatsplats.debug(f"FINAL -> {self._colored_print_opt_params}")

        self.plot_errors(errors_lists)
        v_colours = self.compute_vertex_colours()
        return v_colours, gt_colours, init_colours

    def compute_vertex_colours(self):
        B, V = self.n_sources, self.mesh.N_verts
        v_colours: Float[Tensor, "B V L"] = torch.zeros(
            [B, V, self.kernel_dim],
            device=self.device,
        )

        albo_evals, albo_evecs, mass = self.eigalbo_interp.get_albo_eigenquantities(
            angles=self.angles, scales=self.anisotropies
        )

        kernel_vert_idx = self.mesh.get_face_vertices(self.kernel_face_ids)
        barycentric_coords = self.mesh.cartesian_to_barycentric(
            self.kernel_locations, kernel_vert_idx
        )

        kernel_evecs, kernel_mass = self.eigalbo_interp.barycentric_eig_interpolation(
            eigen_vec=albo_evecs,
            mass=mass,
            barycentric_coords=barycentric_coords,
            vert_idx=kernel_vert_idx,
        )

        v_colours: Float[Tensor, "B V+1 L"] = torch.cat(
            (v_colours, self.kernel_colours.unsqueeze(1)), dim=1
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
            self.diff_times,
        )
        v_colours = v_colours[:V]
        if self.out_net is not None:
            v_colours = self.out_net(v_colours)
        if self.normalize_colours:
            v_colours = utils.normalise_colours(v_colours)
        return v_colours

    @property
    def _colored_print_opt_params(self):
        angles = torch.rad2deg(self.angles).detach().cpu().numpy()
        anisotropies = self.anisotropies.detach().cpu().numpy()
        diff_times = self.diff_times.detach().cpu().numpy()
        kernel_colours = self.kernel_colours.view(-1).detach().cpu().numpy()
        return (
            colored(f"Angles: {angles}, ", "yellow")
            + colored(f"Anisotropies: {anisotropies}, ", "green")
            + colored(f"Diff times: {diff_times}, ", "blue")
            + colored(f"Kernel colours: {kernel_colours}", "red")
        )

    @property
    def kernel_centres(self):
        return self.kernel_locations

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
        pass

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
