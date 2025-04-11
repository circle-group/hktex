from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import heatsplats
from heatsplats.data import MeshSamplerDataModule
from heatsplats.utils.typing import *


from .base import BaseTrainer


@heatsplats.register("trainers.vertex-colours")
class VertexColoursTrainer(BaseTrainer):
    @dataclass
    class Config(BaseTrainer.Config):
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
        vert_idx = data["vert_idx"]
        all_vertices = vert_idx.shape[0] == self.mesh.verts.shape[0]

        albo_weights = self.eigalbo_interp.interpolate_anisotropies(
            angles=self.model.angles, scales=self.model.anisotropies
        )
        albo_evals, albo_evecs, mass = self.eigalbo_interp.albo_vertices(
            albo_weights=albo_weights, vert_idx=None if all_vertices else vert_idx
        )

        data["evals"] = albo_evals
        data["pts_evecs"] = albo_evecs
        data["pts_mass"] = mass
        data["albo_weights"] = albo_weights

        return data
