from dataclasses import dataclass, field
from abc import abstractmethod
import numpy as np
from termcolor import colored
import trimesh
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import hktex

import hktex.utils as utils
from hktex.utils import BaseModule
from hktex.utils.typing import *

__all__ = ["TextureModel"]


class TextureModel(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    scene_min: Float[Tensor, "3"]
    scene_max: Float[Tensor, "3"]

    def configure(
        self,
        **kwargs,
    ):
        super().configure()

        if self.requires_scene_bounds():
            scene_min = kwargs.get("scene_min", [0.0, 0.0, 0.0])
            self.register_buffer(
                "scene_min",
                torch.tensor(scene_min, dtype=torch.float32, device=self.device),
            )
            scene_max = kwargs.get("scene_max", [1.0, 1.0, 1.0])
            self.register_buffer(
                "scene_max",
                torch.tensor(scene_max, dtype=torch.float32, device=self.device),
            )

    def requires_scene_bounds(self) -> Bool:
        return True

    def set_scene_bounds(self, scene_min=None, scene_max=None):
        if scene_min is not None:
            if not torch.is_tensor(scene_min):
                scene_min = torch.tensor(
                    scene_min, dtype=torch.float32, device=self.device
                )
            assert len(scene_min.shape) == 1 and scene_min.shape[0] == 3
            self.scene_min = scene_min.to(self.device)

        if scene_max is not None:
            if not torch.is_tensor(scene_max):
                scene_max = torch.tensor(
                    scene_max, dtype=torch.float32, device=self.device
                )
            assert len(scene_max.shape) == 1 and scene_max.shape[0] == 3
            self.scene_max = scene_max.to(self.device)

    def requires_face_ids(self) -> Bool:
        return False

    @abstractmethod
    def forward(
        self, pts: Float[Tensor, "P in_dim"], **kwargs
    ) -> Float[Tensor, "P out_dim"]:
        pass

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
