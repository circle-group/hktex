from dataclasses import dataclass, field
from abc import abstractmethod
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import trimesh

from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import heatsplats
from heatsplats.modules import GeodesicTracer, GeodesicOpt, EigenAlboInterpolation
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


class OptimiseHeatKernels(BaseObject):
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
        self._logger = logging.getLogger("heatsplats")

        self.n_sources = self.cfg.n_sources
        self.kernel_dim = self.cfg.kernel_dim
        self.normalize_colours = self.cfg.normalize_colours
        self._lrs = self.cfg.lrs
        self._lr_mult = self._lrs.multiplier

        self.datamodule = datamodule
        mesh = self.datamodule.mesh

        self._verts = torch.tensor(mesh.vertices, device=self.device, dtype=torch.float)
        self._faces = torch.tensor(mesh.faces, device=self.device)
        self._fnorms = torch.tensor(mesh.face_normals, device=self.device)

        self.eigalbo_interp = EigenAlboInterpolation(
            self.cfg.eigen_albo, self._verts, self._faces, self._fnorms
        )
        self.tracer: GeodesicTracer = heatsplats.find(self.cfg.tracer_type)(
            self.cfg.tracer,
            self._verts,
            self._faces,
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
            0, self._faces.shape[0], (n_sources,), device=self.device
        )
        kernel_locations = utils.uniform_sample_triangle(
            torch.rand((n_sources, 2), device=self.device)
        )
        # Convert to cartesian coordinates
        kernel_vert_idx = self._faces[kernel_face_ids]
        B, T = kernel_vert_idx.shape
        kernel_vertx = self._verts[kernel_vert_idx.view(B * T)].view(B, T, -1)
        kernel_locations = utils.bary_to_cart_coords(kernel_locations, kernel_vertx)

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

        B, V = self.n_sources, self._verts.shape[0]

        self._logger.info(f"INITIAL -> {self._colored_print_opt_params}")

        errors_lists = {k: [] for k in self._splats.keys()}

        for i in (pbar := tqdm(range(n_iter))):
            data = next(data_iter)
            data = self.prepare_batch(data)

            pos: Tensor = data["pos"]
            gt_colours: Tensor = data["colour"]
            evals: Tensor = data["evals"]
            verts_evecs: Tensor = data["verts_evecs"]
            verts_mass: Tensor = data["verts_mass"]
            pts_evecs: Tensor = data["pts_evecs"]
            pts_mass: Tensor = data["pts_mass"]

            P = pos.shape[0]

            colours: Float[Tensor, "B P L"] = torch.zeros(
                [B, P, self.kernel_dim],
                device=self.device,
            )

            kernel_vert_idx = self._faces[self.kernel_face_ids]
            B, T = kernel_vert_idx.shape
            kernel_vertx = self._verts[kernel_vert_idx.view(B * T)].view(B, T, -1)
            barycentric_coords = utils.cart_to_bary_coords(
                self.kernel_locations, kernel_vertx
            )
            # TODO: Can be cleaned a bit, saved for tracer in optimizer
            setattr(
                self._splats["kernel_locations"],
                "bary_coords",
                barycentric_coords.detach(),
            )

            kernel_evecs, kernel_mass = (
                self.eigalbo_interp.barycentric_eig_interpolation(
                    eigen_vec=verts_evecs,
                    mass=verts_mass,
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
                        self._logger.info(
                            f"{name}: {self._splats[name].grad.data.norm(2)}"
                        )

            for optimizer in self._optims.values():
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                errors = self._errors
                if i == 0 or (i + 1) % 100 == 0:
                    self._logger.info(
                        f"Iteration: {i + 1} -> Loss: {loss.item()}. {errors['printables']}",
                    )

                for k in self._splat_param_keys:
                    if k in errors:
                        errors_lists[k].append(errors[k].item())
                pbar.set_postfix_str(f"Loss: {loss.item():0.4f}")

        self._logger.info(f"FINAL -> {self._colored_print_opt_params}")

        self.plot_errors(errors_lists)
        v_colours = self.compute_vertex_colours()
        return v_colours, gt_colours, init_colours

    def compute_vertex_colours(self):
        B, V = self.n_sources, self._verts.shape[0]
        v_colours: Float[Tensor, "B V L"] = torch.zeros(
            [B, V, self.kernel_dim],
            device=self.device,
        )

        albo_evals, albo_evecs, mass = self.eigalbo_interp.get_albo_eigenquantities(
            angles=self.angles, scales=self.anisotropies
        )

        kernel_vert_idx = self._faces[self.kernel_face_ids]
        B, T = kernel_vert_idx.shape
        kernel_vertx = self._verts[kernel_vert_idx.view(B * T)].view(B, T, -1)
        barycentric_coords = utils.cart_to_bary_coords(
            self.kernel_locations, kernel_vertx
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
        # kernel_vert_idx = self._faces[self.kernel_face_ids]
        # B, T = kernel_vert_idx.shape
        # kernel_vertx = self._verts[kernel_vert_idx.view(B * T)].view(B, T, -1)
        # barycentric_coords = self.kernel_locations
        # return utils.bary_to_cart_coords(barycentric_coords, kernel_vertx)
        return self.kernel_locations

    @property
    @abstractmethod
    def _errors(self):
        pass

    @staticmethod
    @abstractmethod
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


@heatsplats.register("optimise-uv-texture")
class OptimiseUvTexture(OptimiseHeatKernels):
    @dataclass
    class Config(OptimiseHeatKernels.Config):
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

        albo_evals, albo_evecs, mass = self.eigalbo_interp.get_albo_eigenquantities(
            angles=self.angles, scales=self.anisotropies
        )

        data["evals"] = albo_evals
        data["verts_evecs"] = albo_evecs
        data["verts_mass"] = mass

        pts_tri_vert_idx = self._faces[face_ids]  # [P, 3]

        pts_evecs, pts_mass = self.eigalbo_interp.barycentric_eig_interpolation_pts(
            eigen_vec=albo_evecs,
            mass=mass,
            barycentric_coords=barys,
            vert_idx=pts_tri_vert_idx,
        )

        data["pts_evecs"] = pts_evecs
        data["pts_mass"] = pts_mass
        return data

    @property
    def _errors(self):
        return {
            "printables": None,
            "angles": torch.tensor(0),
            "anisotropies": torch.tensor(0),
            "diff_times": torch.tensor(0),
            "kernel_colours": torch.tensor(0),
        }


@heatsplats.register("optimise-vertex-colours")
class OptimiseVertColTexture(OptimiseHeatKernels):
    @dataclass
    class Config(OptimiseHeatKernels.Config):
        pass

    cfg: Config

    def configure(
        self,
        datamodule: MeshSamplerDataModule,
        **kwargs,
    ):
        super().configure(datamodule, **kwargs)

    @property
    def _errors(self):
        return {
            "printables": None,
            "angles": torch.tensor(0),
            "anisotropies": torch.tensor(0),
            "diff_times": torch.tensor(0),
            "kernel_colours": torch.tensor(0),
        }

    def prepare_batch(self, data: dict) -> dict:
        data = super().prepare_batch(data)
        vert_idx = data["vert_idx"]

        albo_evals, albo_evecs, mass = self.eigalbo_interp.get_albo_eigenquantities(
            angles=self.angles, scales=self.anisotropies
        )

        data["evals"] = albo_evals
        data["verts_evecs"] = albo_evecs
        data["verts_mass"] = mass

        # if vert_idx.shape[0] == albo_evecs.shape[1], then all vertices were sampled in normal order
        if vert_idx.shape[0] < albo_evecs.shape[1]:
            data["pts_evecs"] = albo_evecs[:, vert_idx]
            data["pts_mass"] = mass[:, vert_idx]
        else:
            data["pts_evecs"] = data["verts_evecs"]
            data["pts_mass"] = data["verts_mass"]

        return data


@heatsplats.register("optimize-stationary-heat-kernels")
class OptimiseStationaryHeatKernels(OptimiseVertColTexture):
    @dataclass
    class Config(OptimiseHeatKernels.Config):
        gt_source_sampling_method: Optional[str] = "fps"

    cfg: Config

    def configure(
        self,
        datamodule: MeshSamplerDataModule,
        **kwargs,
    ):
        self.cfg.kernel_dim = 3
        self.cfg.lrs.out_net = 0

        super().configure(datamodule, **kwargs)

        assert hasattr(
            datamodule, "bake_heat"
        ), "OptimiseHeatKernelsToKnownStationary requires a datamodule with configure and bake_heat"
        datamodule.configure(
            n_sources=self.n_sources,
            diff_time_scaler_func=self._diff_time_scaler_func,
        )
        datamodule.bake_heat(
            self.eigalbo_interp, self.normalize_colours, device=self.device
        )

    @property
    def gt_splats(self):
        return self.datamodule.gt_splats

    @property
    def _errors(self):
        angles_error = (
            (self.angles.cpu() - torch.deg2rad(self.gt_splats["angles"]))
            .pow(2)
            .sum()
            .pow(0.5)
        )
        anisotropies_error = (
            (self.anisotropies.cpu() - self.gt_splats["anisotropies"])
            .pow(2)
            .sum()
            .pow(0.5)
        )
        diff_times_error = (
            (self.diff_times.cpu() - self.gt_splats["diff_times"]).pow(2).sum().pow(0.5)
        )
        kernel_colours_error = (
            (self.kernel_colours.cpu() - self.gt_splats["kernel_colours"])
            .pow(2)
            .sum()
            .pow(0.5)
        )
        return {
            "printables": (
                "ERRORS: "
                + colored(f"Angles: {angles_error}, ", "yellow")
                + colored(f"Anisotropies: {anisotropies_error}, ", "green")
                + colored(f"Diff times: {diff_times_error}, ", "blue")
                + colored(f"Kernel colours: {kernel_colours_error}", "red")
            ),
            "angles": angles_error,
            "anisotropies": anisotropies_error,
            "diff_times": diff_times_error,
            "kernel_colours": kernel_colours_error,
        }

    @staticmethod
    def plot_errors(errors_lists):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(errors_lists["angles"], label="Angles Error")
        axs[0, 0].set_title("Angles Error")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Error")

        axs[0, 1].plot(errors_lists["anisotropies"], label="Anisotropies Error")
        axs[0, 1].set_title("Anisotropies Error")
        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Error")

        axs[1, 0].plot(errors_lists["diff_times"], label="Diff Times Error")
        axs[1, 0].set_title("Diff Times Error")
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Error")

        axs[1, 1].plot(errors_lists["kernel_colours"], label="Kernel Colours Error")
        axs[1, 1].set_title("Kernel Colours Error")
        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_ylabel("Error")

        for ax in axs.flat:
            ax.legend()

        plt.tight_layout()
        plt.show()


class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


def main(args, extras) -> Dict[str, Any]:
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger = logging.getLogger("heatsplats")
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
            handler.addFilter(ColoredFilter())

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    from heatsplats.utils import ExperimentConfig, load_config, seed_everything

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    seed_everything(cfg.seed)

    datamodule: MeshSamplerDataModule = heatsplats.find(cfg.data_type)(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")

    trainer: OptimiseHeatKernels = heatsplats.find(cfg.trainer_type)(
        cfg.trainer, datamodule
    )

    v_colours, gt_colours, init_colours = trainer.optimise(n_iter=cfg.optim.iters)

    return {
        "optimisation": trainer,
        "datamodule": datamodule,
        "colours": (v_colours, gt_colours, init_colours),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `train.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    args, extras = parser.parse_known_args()
    main(args, extras)
