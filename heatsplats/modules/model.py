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
from .eigen_albo import EigenAlboInterpolation
from .utils import PointsInfo, KernelInfo

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
        mass_type: str = "kde"  # "kde" | "interpolated" | "one"

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

        self._thresholds_act = lambda x: 0.3 + (0.7 - 1e-8) * torch.sigmoid(x + 0.916)
        self._angle_scale = torch.pi
        self._angle_act = lambda x: self._angle_scale * torch.sigmoid(x)
        self._anis_act = lambda x: 1.0 + 99.0 * torch.sigmoid(x)
        self._sharpness_act = lambda x: 10.0 + 40.0 * torch.sigmoid(x)
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

        opacities = torch.rand(self.N_sources, **factory_kwargs) * 2 - 1

        # Initialise angles for uniform output in [0, π/2] considering activation
        p = torch.rand(self.N_sources, **factory_kwargs)
        angles = torch.log(p / (1 - p + 1e-7))

        epsilon = 1e-8
        # Initialize Sharpnesses for uniform output in [10.0, 50.0] considering activation
        random_power = torch.rand(self.N_sources, **factory_kwargs) ** 0.5
        uniform_sharpnesses = 10.0 + 40.0 * random_power
        p_sharp = (uniform_sharpnesses - 10.0) / (50.0 - 10.0)
        sharpnesses = torch.log(p_sharp / (1 - p_sharp + epsilon))

        # Initialize Thresholds for uniform output in [0.3, 0.9999] considering activation
        random_power = torch.rand(self.N_sources, **factory_kwargs) ** 0.7
        uniform_thresholds = 0.3 + (0.7 - 1e-8) * random_power
        p_thresh = (uniform_thresholds - 0.3) / (0.7 - 1e-8)
        thresholds = torch.log(p_thresh / (1 - p_thresh + epsilon)) - 0.916

        # Initialize Anisotropies for uniform output in a chosen range [1, 100]
        uniform_anisotropies = 1.0 + 99.0 * torch.rand(self.N_sources, **factory_kwargs)
        p_anisotropy = (uniform_anisotropies - 1.0) / 99.0
        anisotropies = torch.log(p_anisotropy / (1 - p_anisotropy + epsilon))

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

    def prepare_points_for_diffusion(
        self,
        mesh: Mesh,
        eigalbo_interp: EigenAlboInterpolation,
        albo_weights: Float[Tensor, "G M"],
        face_ids: Float[Tensor, "P"],
        barys: Float[Tensor, "P 3"] | None = None,
        pts: Float[Tensor, "P 3"] | None = None,
    ) -> PointsInfo:
        pts_tri_vert_idx = mesh.get_face_vertices(face_ids)  # [P, 3]

        if barys is None and pts is not None:
            barys = mesh.cartesian_to_barycentric(pts, pts_tri_vert_idx)
        elif barys is not None and pts is None:
            pass
        else:
            raise ValueError(
                "Either barys or pts must be provided to prepare points for diffusion"
            )

        # iso_evecs=None if 'distance_weighting' == "none" in eigalbo_interp config
        iso_evecs = eigalbo_interp.barycentric_ilbo_evec_points(barys, pts_tri_vert_idx)

        albo_evals, pts_evecs, pts_mass = eigalbo_interp.barycentric_albo_points(
            albo_weights=albo_weights,
            barycentric_coords=barys,
            vert_idx=pts_tri_vert_idx,
        )

        return {
            "iso_evecs": iso_evecs,  # [P, K] or None
            "albo_evals": albo_evals,  # [G, K]
            "albo_evecs": pts_evecs,  # [G, P, K]
            "mass": pts_mass,  # [1, P]
        }

    def prepare_kernels_for_diffusion(
        self,
        mesh: Mesh,
        eigalbo_interp: EigenAlboInterpolation,
        albo_weights: Float[Tensor, "G M"],
        save_barycentric: bool = True,
    ) -> KernelInfo:
        kernel_vert_idx = mesh.get_face_vertices(self.kernel_face_ids)
        kernel_barycentric_coords = mesh.cartesian_to_barycentric(
            self.kernel_locations, kernel_vert_idx
        )

        if save_barycentric:
            self.save_barycentric_locations(kernel_barycentric_coords)

        kernel_evecs, kernel_mass = eigalbo_interp.barycentric_albo_gaussians(
            albo_weights=albo_weights,
            barycentric_coords=kernel_barycentric_coords,
            vert_idx=kernel_vert_idx,
        )
        return {
            "vert_idx": kernel_vert_idx,  # [G, 3]
            "barycentric_coords": kernel_barycentric_coords,  # [G, 3]
            "albo_evecs": kernel_evecs,  # [G, K]
            "mass": kernel_mass,  # [G]
        }

    def diffuse_heat_kernels(
        self,
        eigalbo_interp: EigenAlboInterpolation,
        pts_info: PointsInfo,
        kernel_info: KernelInfo,
        at_vertices: bool = False,
    ) -> Float[Tensor, "P D"]:

        pts_iso_evecs: Optional[Float[Tensor, "P K"]] = pts_info["iso_evecs"]
        pts_evecs: Float[Tensor, "G P K"] = pts_info["albo_evecs"]
        pts_evals: Float[Tensor, "G K"] = pts_info["albo_evals"]
        pts_mass: Float[Tensor, "1 P"] | None = pts_info["mass"]

        kernel_vert_idx: Float[Tensor, "G 3"] = kernel_info["vert_idx"]
        kernel_bary: Float[Tensor, "G 3"] = kernel_info["barycentric_coords"]
        kernel_evecs: Float[Tensor, "G K"] = kernel_info["albo_evecs"]
        kernel_mass: Float[Tensor, "G"] | None = kernel_info["mass"]

        G, P = self.N_sources, pts_evecs.shape[1]

        diracs = torch.zeros([G, P, 1], device=pts_evecs.device)
        diracs: Float[Tensor, "G P+1 L"] = torch.cat(
            (diracs, torch.ones([G, 1, 1], device=pts_evecs.device)), dim=1
        )

        pts_evecs: Float[Tensor, "G P+1 K"] = torch.cat(
            ((pts_evecs, kernel_evecs.unsqueeze(1))), dim=1
        )

        pts2kernel_dist: Optional[Float[Tensor, "G P"]] = (
            eigalbo_interp.compute_pts2kernel_biharmonic_distance(
                pts_iso_evecs, kernel_bary, kernel_vert_idx
            )
        )

        # PS: pts_iso_evecs = None and biharmonic_dist_weights = None
        # if 'distance_weighting' == "none" in eigalbo_interp config
        biharmonic_dist_weights = eigalbo_interp.compute_biharmonic_weights(
            pts_iso_evecs, kernel_bary, kernel_vert_idx, pts2kernel_dist
        )

        if self.cfg.mass_type == "kde":
            pts_mass: Float[Tensor, "G P+1"] = (
                eigalbo_interp.compute_biharmonic_dist_kde_mass(
                    pts_iso_evecs,
                    kernel_bary,
                    kernel_vert_idx,
                    pts2kernel_dist,
                    sigma=0.05,
                    total_area_normalise=True,
                )
            )
        elif self.cfg.mass_type == "interpolated":
            pts_mass: Float[Tensor, "G P+1"] = torch.cat(
                (pts_mass.expand(G, -1), kernel_mass.unsqueeze(-1)), dim=1
            )
        elif self.cfg.mass_type == "one":
            # pts_mass: Float[Tensor, "G P+1"] = torch.ones_like(pts_evecs[:, :, 0])
            pts_mass = torch.tensor(1.0, device=pts_evecs.device)
        else:
            raise ValueError(f"Unknown mass type: {self.cfg.mass_type}")

        diffused_diracs: Float[Tensor, "G P+1 1"] = utils.heat_diffusion(
            diracs,
            pts_mass,
            pts_evals,
            pts_evecs,
            self.diff_times,
            biharmonic_dist_weights,
            at_vertices=at_vertices,
        )

        diffused_diracs: Float[Tensor, "G P+1 1"] = diffused_diracs / (
            diffused_diracs[:, P, :].unsqueeze(1) + 1e-8
        )
        diffused_diracs: Float[Tensor, "G P 1"] = diffused_diracs[:, :P, :]

        filtered: Float[Tensor, "G P 1"] = self.kernel_filter_func(
            diffused_diracs,
            epsilon=self.thresholds,
            sharpness=self.sharpnesses,
        )

        contribs: Float[Tensor, "G P 1"] = filtered * self.opacities.view(-1, 1, 1)
        colours: Float[Tensor, "G P D"] = contribs * self.kernel_colours.unsqueeze(1)
        colours: Float[Tensor, "P D"] = colours.sum(dim=0)

        return colours, contribs

    def forward(self, x_diffusion: Float[Tensor, "P D"]) -> Float[Tensor, "P out_dim"]:
        out = x_diffusion
        if self.out_net is not None:
            out = self.out_net(out)
        if self.normalize_colours:
            out = utils.normalise_colours(out)
        return out

    def compute_vertex_colours(
        self, mesh: Mesh, eigalbo_interp: EigenAlboInterpolation
    ):
        albo_weights = eigalbo_interp.interpolate_anisotropies(
            angles=self.angles, scales=self.anisotropies
        )
        albo_evals, albo_evecs, mass = eigalbo_interp.albo_vertices(
            albo_weights=albo_weights
        )
        ilbo_evecs = eigalbo_interp.ilbo_evec_vertices()

        verts_info: PointsInfo = {
            "iso_evecs": ilbo_evecs,  # [V, K] or None
            "albo_evals": albo_evals,  # [G, K]
            "albo_evecs": albo_evecs,  # [G, V, K]
            "mass": mass,  # [1, V]
        }

        kernel_info: KernelInfo = self.prepare_kernels_for_diffusion(
            mesh=mesh,
            eigalbo_interp=eigalbo_interp,
            albo_weights=albo_weights,
            save_barycentric=False,
        )

        v_colours, _ = self.diffuse_heat_kernels(
            eigalbo_interp=eigalbo_interp,
            pts_info=verts_info,
            kernel_info=kernel_info,
            at_vertices=True,
        )  # [V, D]

        v_colours = self.forward(v_colours)  # Postprocess

        return v_colours

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
