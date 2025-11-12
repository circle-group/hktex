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

import tinycudann as tcnn

import heatsplats

import heatsplats.utils as utils
from heatsplats.modules.base import TextureNetwork
from heatsplats.utils.typing import *


__all__ = ["MLPTextureNetwork"]


@heatsplats.register("modules.mlp-texture-network")
class MLPTextureNetwork(TextureNetwork):
    @dataclass
    class Config(TextureNetwork.Config):
        input_dim: int = 3
        output_dim: int = 3

        width: int = 128
        hidden: int = 5

        encoding: dict[str, Any] = field(default_factory=dict)

    cfg: Config

    def configure(
        self,
        **kwargs,
    ):
        super().configure()

        encoding_config = OmegaConf.to_container(self.cfg.encoding)
        self.encoding = tcnn.Encoding(
            self.cfg.input_dim, encoding_config, dtype=torch.float32
        ).to(self.device)
        self.encoding_dim = self.encoding.n_output_dims

        in_size = self.cfg.input_dim + self.encoding_dim
        out_size = self.cfg.output_dim
        width = self.cfg.width

        hidden_layers = []
        for i in range(self.cfg.hidden):
            hidden_layers.extend([nn.Linear(width, width), nn.LeakyReLU(inplace=True)])

        self.network = nn.Sequential(
            nn.Linear(in_size, width),
            nn.LeakyReLU(inplace=True),
            *hidden_layers,
            nn.Linear(width, out_size),
            nn.Sigmoid(),
        ).to(self.device)

    def _preprocess(self, pts: Float[Tensor, "P in_dim"]) -> Float[Tensor, "P in_dim"]:
        pts = (pts - self.scene_min) / (self.scene_max - self.scene_min)
        pts = 2 * pts + 1
        return pts

    def forward(
        self, pts: Float[Tensor, "P in_dim"], **kwargs
    ) -> Float[Tensor, "P out_dim"]:
        pts = self._preprocess(pts)
        net_in = torch.cat((pts, self.encoding(pts)), dim=-1)
        out = self.network(net_in)
        return out
