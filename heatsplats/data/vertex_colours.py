from dataclasses import dataclass
import trimesh

import torch
from torch.utils.data import IterableDataset, DataLoader, default_collate

import heatsplats
from heatsplats.utils import parse_structured
from heatsplats.utils.typing import *

from .base import MeshSamplerDataModule, MeshSamplerDataConfig


@dataclass
class VertexColoursDataConfig(MeshSamplerDataConfig):
    sample_all_vertices: bool = False

    batch_size: int = 128
    num_workers: int = 4


class VertexColoursDataset(IterableDataset):
    def __init__(self, cfg: VertexColoursDataConfig, mesh: trimesh.Trimesh, split: str):
        super().__init__()

        self.cfg = cfg
        self.mesh = mesh
        self.split = split

        self.verts = torch.tensor(mesh.vertices)
        self.vcols = self.get_vertex_colours()

        self.sample_all_vertices = self.cfg.sample_all_vertices

    def get_vertex_colours(self):
        try:
            self.mesh.visual.vertex_colors
        except AttributeError:
            self.mesh.visual = self.mesh.visual.to_color()
        return torch.tensor(
            self.mesh.visual.vertex_colors[:, :3] / 255, dtype=torch.float
        )

    def __iter__(self):
        while True:
            verts, vcols = self.verts, self.vcols
            if self.sample_all_vertices:
                vert_idx = torch.arange(self.vcols.shape[0])
            else:
                vert_idx = torch.randint(0, self.vcols.shape[0], (self.cfg.batch_size,))

                verts = verts[vert_idx]
                vcols = vcols[vert_idx]
            yield {"pos": verts, "colour": vcols, "vert_idx": vert_idx}


@heatsplats.register("data.vertex-colours")
class VertexColoursDataModule(MeshSamplerDataModule):
    cfg: VertexColoursDataConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        cfg = parse_structured(VertexColoursDataConfig, cfg)

        super().__init__(cfg=cfg)

    def setup(self, stage=None) -> None:
        super().setup(stage)
        if stage in [None, "fit"]:
            self.train_dataset = VertexColoursDataset(self.cfg, self.mesh, "train")
