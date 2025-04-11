from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import heatsplats
from heatsplats.data import MeshSamplerDataModule
from heatsplats.utils.typing import *

from .base import BaseTrainer


@heatsplats.register("trainers.uv-texture")
class UvTextureTrainer(BaseTrainer):
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
        face_ids = data["face_id"]
        barys = data["bary"]

        pts_tri_vert_idx = self.mesh.get_face_vertices(face_ids)  # [P, 3]

        albo_evals, pts_evecs, pts_mass, albo_weights = (
            self.eigalbo_interp.barycentric_albo_eigenquantities(
                angles=self.angles,
                scales=self.anisotropies,
                barycentric_coords=barys,
                vert_idx=pts_tri_vert_idx,
            )
        )

        data["evals"] = albo_evals
        data["pts_evecs"] = pts_evecs
        data["pts_mass"] = pts_mass
        data["albo_weights"] = albo_weights

        return data
