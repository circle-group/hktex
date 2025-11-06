from dataclasses import dataclass

import torch
import trimesh

import heatsplats
from heatsplats.utils import BaseObject, load_mesh
from heatsplats.utils.typing import *

from .importance_sampling_strategy import (
    ImportanceSamplingMixIn,
    ImportanceSamplingDataConfigMixIn,
)


@dataclass
class MeshSamplerDataConfig(ImportanceSamplingDataConfigMixIn):
    mesh_path: str = "../objects/spot/spot_triangulated.py"
    bake_vert_colours_if_textured: bool = True
    merge_tex: bool = False
    use_importance_sampling: bool = False


class MeshSamplerDataModule(ImportanceSamplingMixIn):
    cfg: MeshSamplerDataConfig

    mesh: trimesh.Trimesh

    def __init__(self, cfg: MeshSamplerDataConfig):
        self.cfg = cfg

    def load_mesh(self):
        mesh_path = self.cfg.mesh_path
        has_texture = mesh_path.endswith((".glb", ".obj"))
        bake_vert_colours = has_texture and self.cfg.bake_vert_colours_if_textured
        self.mesh = load_mesh(
            mesh_path,
            show=False,
            merge_tex=self.cfg.merge_tex,
            bake_vert_colors=bake_vert_colours,
        )

    def prepare_data(self):
        self.load_mesh()

    def setup(self, stage=None) -> None:
        assert hasattr(
            self, "mesh"
        ), "Please call prepare_data before setup in data module"
