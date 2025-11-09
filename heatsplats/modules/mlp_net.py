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
from heatsplats.utils import BaseModule
from heatsplats.utils.typing import *


__all__ = ["MLPNetwork"]


class MLPNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
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

    def forward(self, pts: Float[Tensor, "P in_dim"]) -> Float[Tensor, "P out_dim"]:
        net_in = torch.cat((pts, self.encoding(pts)), dim=-1)
        out = self.network(net_in)
        return out

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
