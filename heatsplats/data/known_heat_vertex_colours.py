from dataclasses import dataclass
import trimesh
import logging
from termcolor import colored

import torch
from torch.utils.data import DataLoader

import heatsplats
import heatsplats.utils as utils
from heatsplats.modules import EigenAlboInterpolation
from heatsplats.utils import parse_structured, farthest_point_sampling, heat_diffusion
from heatsplats.utils.typing import *

from .base import MeshSamplerDataModule
from .vertex_colours import VertexColoursDataset, VertexColoursDataConfig


@dataclass
class KnownHeatVertexColoursDataConfig(VertexColoursDataConfig):
    source_sampling_method: str = "fps"


@heatsplats.register("data.known-heat-vertex-colours")
class KnownHeatVertexColoursDataModule(MeshSamplerDataModule):
    cfg: KnownHeatVertexColoursDataConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        cfg = parse_structured(KnownHeatVertexColoursDataConfig, cfg)

        super().__init__(cfg=cfg)

    def setup(self, stage=None) -> None:
        super().setup(stage)
        if stage in [None, "fit"]:
            self.train_dataset = VertexColoursDataset(self.cfg, self.mesh, "train")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=None,  # Disable automatic batching
            num_workers=self.cfg.num_workers,
        )

    def configure(self, n_sources, diff_time_scaler_func):
        assert hasattr(self, "train_dataset"), "Please call setup before configure"
        source_sampling_method = self.cfg.source_sampling_method
        self.n_sources = n_sources

        if n_sources == 3:
            self.source_idxs = torch.tensor([992, 1150, 875])

            self.gt_splats = {
                "angles": torch.tensor([45.0, 18.3, 10.0]),
                "anisotropies": torch.tensor([33.0, 60, 5.2]),
                "diff_times": torch.tensor([0.001, 0.1, 0.01]),
                "kernel_colours": torch.tensor(
                    [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                    dtype=torch.float,
                ),
            }
        else:
            if source_sampling_method == "fps":
                source_idx = farthest_point_sampling(
                    self.train_dataset.verts, n_sources
                )
                self.source_idxs = source_idx.nonzero(as_tuple=True)[0]
            else:
                self.source_idxs = torch.randint(
                    0, len(self.train_dataset.verts), (n_sources,)
                )

            self.gt_splats = {
                "angles": torch.rand(n_sources) * 180,
                "anisotropies": (100 * torch.rand(n_sources)),
                "diff_times": diff_time_scaler_func(torch.rand(n_sources)),
                "kernel_colours": torch.rand(
                    (n_sources, 3),
                    dtype=torch.float,
                ),
            }

    def bake_heat(
        self,
        eigalbo_interp: EigenAlboInterpolation,
        normalise_colours: bool = False,
        device=None,
    ):
        idx_range = torch.arange(self.n_sources, device=device)

        gt_colours = torch.zeros(
            [self.n_sources, *self.train_dataset.verts.shape], device=device
        )
        gt_colours[idx_range, self.source_idxs, :] = self.gt_splats[
            "kernel_colours"
        ].to(device)

        albo_weights = eigalbo_interp.interpolate_anisotropies(
            angles=torch.deg2rad(self.gt_splats["angles"].to(device)),
            scales=self.gt_splats["anisotropies"].to(device),
        )
        albo_evals, albo_evecs, mass = eigalbo_interp.albo_vertices(
            albo_weights=albo_weights
        )

        gt_colours = heat_diffusion(
            gt_colours.to(device),
            mass,
            albo_evals,
            albo_evecs,
            self.gt_splats["diff_times"].to(device),
        )

        gta = self.gt_splats["angles"].detach().cpu().numpy()
        gts = self.gt_splats["anisotropies"].detach().cpu().numpy()
        gtt = self.gt_splats["diff_times"].detach().cpu().numpy()
        gtc = self.gt_splats["kernel_colours"].detach().cpu().numpy()

        logger = logging.getLogger("heatsplats")
        logger.info(
            "GT -> %s %s %s %s",
            colored(f"Angles: {gta}, ", "yellow"),
            colored(f"Anisotropies: {gts}, ", "green"),
            colored(f"Diff times: {gtt}, ", "blue"),
            colored(f"Kernel colours: {gtc}", "red"),
        )

        gt_colours = gt_colours.sum(dim=0)
        if normalise_colours:
            gt_colours = utils.normalise_colours(gt_colours)

        self.train_dataset.vcols = gt_colours.detach().cpu()
