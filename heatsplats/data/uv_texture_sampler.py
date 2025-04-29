from dataclasses import dataclass
import trimesh

import torch
from torch.utils.data import IterableDataset, DataLoader

import heatsplats
from heatsplats.utils import (
    parse_structured,
    uniform_sample_triangle,
    interpolate_barycentric_attr,
    uniform_sampling,
)
from heatsplats.utils.typing import *

from .base import MeshSamplerDataModule, MeshSamplerDataConfig


@dataclass
class UvTextureSamplerDataConfig(MeshSamplerDataConfig):
    batch_size: int = 128
    num_workers: int = 4
    sampling_method: str = "uniform"  # "random" or "uniform"


class UvTextureSamplerDataset(IterableDataset):
    def __init__(
        self, cfg: UvTextureSamplerDataConfig, mesh: trimesh.Trimesh, split: str
    ):
        super().__init__()

        self.cfg = cfg
        self.mesh = mesh
        self.split = split

        self.verts = torch.tensor(mesh.vertices)
        self.faces = torch.tensor(mesh.faces)

        self.uv = torch.tensor(self.mesh.visual.uv)

        if hasattr(mesh, "original_faces") and hasattr(mesh, "original_uv"):
            self.tex_faces = torch.tensor(mesh.original_faces)
            self.uv = torch.tensor(mesh.original_uv)
        else:
            self.tex_faces = self.faces

        self.tex_img = self.get_texture_image()

    def get_texture_image(self):
        try:
            tex_img = self.mesh.visual.material.baseColorTexture
            if tex_img is None:
                raise AttributeError
        except AttributeError:
            tex_img = self.mesh.visual.material.image
            if tex_img is None:
                raise AttributeError
        return tex_img

    def __iter__(self):
        while True:
            batch_size = self.cfg.batch_size

            if self.cfg.sampling_method == "random":
                face_id = torch.randint(0, self.faces.shape[0], (batch_size,))
                bary_coord = uniform_sample_triangle(torch.rand((batch_size, 2)))
            elif self.cfg.sampling_method == "uniform":
                face_id, bary_coord = uniform_sampling(
                    self.verts, self.faces, batch_size
                )
            else:
                raise ValueError(f"Unknown sampling method: {self.cfg.sampling_method}")

            pos = interpolate_barycentric_attr(
                self.faces, face_id, bary_coord, self.verts
            )
            uv = interpolate_barycentric_attr(
                self.tex_faces, face_id, bary_coord, self.uv
            )
            color = trimesh.visual.uv_to_color(uv, self.tex_img)[:, :3] / 255
            color = torch.tensor(color, dtype=torch.float)
            yield {"pos": pos, "colour": color, "face_id": face_id, "bary": bary_coord}


@heatsplats.register("data.uv-texture-sampler")
class UvTextureSamplerDataModule(MeshSamplerDataModule):
    cfg: UvTextureSamplerDataConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        cfg = parse_structured(UvTextureSamplerDataConfig, cfg)

        super().__init__(cfg=cfg)

    def setup(self, stage=None) -> None:
        super().setup(stage)
        if stage in [None, "fit"]:
            self.train_dataset = UvTextureSamplerDataset(self.cfg, self.mesh, "train")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=None,  # Disable automatic batching
            num_workers=self.cfg.num_workers,
        )
