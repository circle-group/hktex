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
