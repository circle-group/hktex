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
        diff_time: float = 1e-2

    cfg: Config

    _kernel_colours: Float[Tensor, "G D"]
    _angles: Float[Tensor, "G"]
    _anisotropies: Float[Tensor, "G"]
    _thresholds: Float[Tensor, "G"]
    _sharpnesses: Float[Tensor, "G"]
    _opacities: Float[Tensor, "G"]
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

        self._thresholds_act = lambda x: 0.1 + (0.9 - 1e-8) * torch.sigmoid(x)
        self._angle_scale = torch.pi / 2
        self._angle_act = lambda x: self._angle_scale * torch.sigmoid(x)
        self._anis_act = lambda x: torch.exp(x)  # TODO: why exp?
        self._sharpness_act = lambda x: 5.0 + 95.0 * torch.sigmoid(x)
        self._opacity_act = lambda x: torch.clamp(x, min=-1, max=1)
        self._colour_act = lambda x: x

        self.kernel_filter_func = utils.rescaled_soft_step

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
        factory_kwargs = {"dtype": torch.float, "device": self.device}
        if self.out_net is not None:
            kernel_colours = torch.randn(
                (self.N_sources, self.kernel_dim), **factory_kwargs
            )
        else:
            kernel_colours = torch.rand(
                (self.N_sources, self.out_dim), **factory_kwargs
            )

        opacities = torch.rand(self.N_sources, **factory_kwargs)

        # Initialise angles for uniform output in [0, π/2] considering activation
        p = torch.rand(self.N_sources, **factory_kwargs)
        angles = torch.log(p / (1 - p + 1e-7))

        epsilon = 1e-5
        # Initialize Sharpnesses for uniform output in [5.0, 100.0] considering activation
        uniform_sharpnesses = 5.0 + 45.0 * torch.rand(self.N_sources, **factory_kwargs)
        p_sharp = (uniform_sharpnesses - 5.0) / 95.0
        sharpnesses = torch.log(p_sharp / (1 - p_sharp + epsilon))

        # Initialize Thresholds for uniform output in [0.1, 0.9999] considering activation
        uniform_thresholds = 0.25 + 0.6 * torch.rand(self.N_sources, **factory_kwargs)
        p_thresh = (uniform_thresholds - 0.1) / (0.9 - 1e-8)
        thresholds = torch.log(p_thresh / (1 - p_thresh + epsilon))

        # Initialize Anisotropies for uniform output in a chosen range [min_val, max_val]
        min_val, max_val = 0.1, 100.0
        uniform_anisotropies = min_val + (max_val - min_val) * torch.rand(
            self.N_sources, **factory_kwargs
        )
        anisotropies = torch.log(uniform_anisotropies)

        # Uniformly sample many points on the mesh surface
        # and then use farthest point sampling to select the kernel locations
        fids, bary = utils.uniform_sampling(
            mesh.verts, mesh.faces, max(3 * self.N_sources, 10_000)
        )
        pos = mesh.barycentric_to_cartesian(bary, mesh.get_face_vertices(fids))
        mask = utils.farthest_point_sampling(pos, self.N_sources)
        kernel_locations = pos[mask]
        kernel_face_ids = fids[mask]

        self._kernel_colours = torch.nn.Parameter(kernel_colours)
        self._angles = torch.nn.Parameter(angles)
        self._anisotropies = torch.nn.Parameter(anisotropies)
        self._thresholds = torch.nn.Parameter(thresholds)
        self._sharpnesses = torch.nn.Parameter(sharpnesses)
        self._opacities = torch.nn.Parameter(opacities)
        self._kernel_locations = nn.Parameter(kernel_locations)
        self._kernel_face_ids = nn.Buffer(kernel_face_ids, persistent=True)

        self.splat_param_keys = [
            "kernel_colours",
            "angles",
            "anisotropies",
            "sharpnesses",
            "thresholds",
            "opacities",
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
        return self.cfg.diff_time * torch.ones(self.N_sources, device=self.device)

    @property
    def thresholds(self) -> Float[Tensor, "G"]:
        return self._thresholds_act(self._thresholds)

    @property
    def sharpnesses(self) -> Float[Tensor, "G"]:
        return self._sharpness_act(self._sharpnesses)

    @property
    def opacities(self) -> Float[Tensor, "G"]:
        return self._opacity_act(self._opacities)

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
            + colored(f"Kernel colours: {kernel_colours}", "red")
            + colored(f"Opacities: {self.opacities}", "magenta")
            + colored(f"Sharpnesses: {self.sharpnesses}", "cyan")
            + colored(f"Thresholds: {self.thresholds}", "blue")
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
