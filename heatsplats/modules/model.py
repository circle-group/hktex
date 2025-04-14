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

import heatsplats.utils as utils
from heatsplats.utils import BaseModule
from heatsplats.utils.typing import *

from .mesh import Mesh

__all__ = ["Model"]


class Model(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        n_sources: int = 128
        out_dim: int = 3

        kernel_dim: int = 32
        out_net: bool = True
        normalize_colours: bool = False

    cfg: Config

    _kernel_colours: Float[Tensor, "G D"]
    _angles: Float[Tensor, "G"]
    _anisotropies: Float[Tensor, "G"]
    _diff_times: Float[Tensor, "G"]
    _kernel_locations: Float[Tensor, "G 3"]
    _kernel_face_ids: Int[Tensor, "G"]

    def configure(
        self,
        mesh: Mesh,
        **kwargs,
    ):
        super().configure()

        self.N_sources = self.cfg.n_sources
        self.out_dim = self.cfg.out_dim

        self.kernel_dim = self.cfg.kernel_dim
        self.normalize_colours = self.cfg.normalize_colours

        self._diff_time_scaler_func = lambda x: 10 ** (4 * torch.tanh(x) - 2)
        self._angle_scaler = torch.pi
        self._angle_act = lambda x: self._angle_scaler * F.hardsigmoid(x)
        self._anis_act = lambda x: torch.exp(x)
        self._colour_act = lambda x: x

        self.out_net = None
        if self.cfg.out_net:
            self.out_net = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.kernel_dim, 2 * self.kernel_dim),
                nn.ReLU(),
                nn.Linear(2 * self.kernel_dim, self.out_dim),
                nn.Sigmoid(),
            ).to(self.device)

        self._make_splats(mesh)

    def _make_splats(self, mesh: Mesh):
        kernel_colours = torch.randn(
            (self.N_sources, self.kernel_dim), dtype=torch.float, device=self.device
        )
        angles = torch.randn(self.N_sources, device=self.device)
        anisotropies = torch.randn(self.N_sources, device=self.device)
        diff_times = torch.rand(self.N_sources, device=self.device)

        # Sample face and barycentric location on face
        kernel_face_ids = torch.randint(
            0, mesh.N_faces, (self.N_sources,), device=self.device
        )
        kernel_locations = utils.uniform_sample_triangle(
            torch.rand((self.N_sources, 2), device=self.device)
        )
        # Convert to cartesian coordinates
        kernel_vert_idx = mesh.get_face_vertices(kernel_face_ids)
        kernel_locations = mesh.barycentric_to_cartesian(
            kernel_locations, kernel_vert_idx
        )

        self._kernel_colours = torch.nn.Parameter(kernel_colours)
        self._angles = torch.nn.Parameter(angles)
        self._anisotropies = torch.nn.Parameter(anisotropies)
        self._diff_times = torch.nn.Parameter(diff_times)
        self._kernel_locations = nn.Parameter(kernel_locations)
        self._kernel_face_ids = nn.Buffer(kernel_face_ids, persistent=True)

        self.splat_param_keys = [
            "kernel_colours",
            "angles",
            "anisotropies",
            "diff_times",
        ]

    @property
    def kernel_colours(self) -> Float[Tensor, "G D"]:
        return self._colour_act(self._kernel_colours)

    @property
    def angles(self) -> Float[Tensor, "G"]:
        return self._angle_act(self._angles)

    @property
    def anisotropies(self) -> Float[Tensor, "G"]:
        return self._anis_act(self._anisotropies)

    @property
    def diff_times(self) -> Float[Tensor, "G"]:
        return self._diff_time_scaler_func(self._diff_times)

    @property
    def kernel_locations(self) -> Float[Tensor, "G 3"]:
        return self._kernel_locations

    @property
    def kernel_face_ids(self) -> Int[Tensor, "G"]:
        return self._kernel_face_ids

    def forward(self, x_diffusion: Float[Tensor, "P D"]) -> Float[Tensor, "P out_dim"]:
        out = x_diffusion
        if self.out_net is not None:
            out = self.out_net(out)
        if self.normalize_colours:
            out = utils.normalise_colours(out)
        return out

    @property
    def colored_print_opt_params(self):
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

    def save_barycentric_locations(self, barycentric_coords):
        setattr(
            self._kernel_locations,
            "bary_coords",
            barycentric_coords.detach(),
        )

    def save_torch(self, filename):
        torch.save(self.state_dict(), filename)

    def save_numpy_npz(self, filename):
        np_dict = {}
        for k, v in self.state_dict().items():
            np_dict[k] = v.detach().cpu().numpy()
        np.savez_compressed(filename, **np_dict)

    def load_torch(self, filename):
        self.load_state_dict(
            torch.load(filename, map_location=self.device, weights_only=True)
        )
