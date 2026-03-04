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

try:
    import tinycudann as tcnn

    HAS_TCNN = True
except ImportError:
    HAS_TCNN = False
    tcnn = None

import heatsplats

import heatsplats.utils as utils
from heatsplats.modules import Mesh, EigenAlboInterpolation
from heatsplats.modules.base import TextureModel
from heatsplats.utils.typing import *


__all__ = ["MLPTextureNetwork"]


class PositionalEmbedding(nn.Module):
    def __init__(self, n_freqs: int = 6, include_input: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        self.out_dim = (3 * 2 * n_freqs) + (3 if include_input else 0)
        self.register_buffer(
            "freq_bands", 2.0 ** torch.linspace(0.0, n_freqs - 1, n_freqs)
        )

    @property
    def output_dim(self):
        return self.out_dim

    def forward(self, x, **kwargs):
        outs = [x] if self.include_input else []
        for freq in self.freq_bands:
            outs.append(torch.sin(x * freq * torch.pi))
            outs.append(torch.cos(x * freq * torch.pi))
        return torch.cat(outs, dim=-1)


class LaplacianEmbedding(nn.Module):
    def __init__(self, mesh, cfg: dict):
        super().__init__()
        self.mesh: Mesh = mesh

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        else:
            cfg = cfg.copy()

        self.effective_k_eig = cfg.pop("effective_k_eig", None)

        ea_type = cfg.pop("eigen_albo_type", "modules.eigen-albo-interpolation")
        EigenAlboClass = heatsplats.find(ea_type)

        if hasattr(EigenAlboClass, "Config"):
            valid_keys = set(OmegaConf.structured(EigenAlboClass.Config).keys())
            cfg = {k: v for k, v in cfg.items() if k in valid_keys}

        self.eigalbo: EigenAlboInterpolation = EigenAlboClass(cfg, mesh)

        if self.effective_k_eig is not None:
            self.out_dim = self.effective_k_eig
        else:
            self.out_dim = self.eigalbo.cfg.k_eig

    @property
    def output_dim(self):
        return self.out_dim

    def forward(self, pts, **kwargs):
        face_ids = kwargs.get("face_ids")
        if face_ids is None:
            raise ValueError("LaplacianEmbedding requires 'face_ids' in kwargs.")

        vert_ids = self.mesh.get_face_vertices(face_ids.int())

        barys = kwargs.get("barys")
        if barys is None:
            barys = self.mesh.cartesian_to_barycentric(pts, vert_ids)

        evecs = self.eigalbo._iso_eigen_vec
        if self.effective_k_eig is not None:
            evecs = evecs[:, : self.effective_k_eig]

        return utils.interpolate_barycentric_attr_from_trivertidx(
            vert_ids, barys, evecs
        )


@heatsplats.register("modules.mlp-texture-network")
class MLPTextureNetwork(TextureModel):
    @dataclass
    class Config(TextureModel.Config):
        input_dim: int = 3
        output_dim: int = 3

        width: int = 128
        hidden: int = 5

        encoding_type: str = "hash"  # "hash", "laplacian", "positional"
        encoding: dict[str, Any] = field(default_factory=dict)

    cfg: Config

    def configure(
        self,
        **kwargs,
    ):
        super().configure()

        self.mesh = kwargs.get("mesh", None)

        self.encoding_model = self._create_encoding()

        if self.cfg.encoding_type == "hash":
            in_size = self.cfg.input_dim + self.encoding_model.output_dim
        else:
            in_size = self.encoding_model.output_dim

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

    def _create_encoding(self):
        if self.cfg.encoding_type == "hash":
            if not HAS_TCNN:
                raise ImportError(
                    "tinycudann required for MLP Textures with hash encoding"
                )
            encoding_config = OmegaConf.to_container(self.cfg.encoding)

            class TCNNWrapper(nn.Module):
                def __init__(self, encoding):
                    super().__init__()
                    self.encoding = encoding
                    n_params = sum(p.numel() for p in self.encoding.parameters())

                @property
                def output_dim(self):
                    return self.encoding.n_output_dims

                def forward(self, x, **kwargs):
                    return self.encoding(x)

            return TCNNWrapper(
                tcnn.Encoding(
                    self.cfg.input_dim, encoding_config, dtype=torch.float32
                ).to(self.device)
            )

        elif self.cfg.encoding_type == "laplacian":
            if self.mesh is None:
                raise ValueError("Mesh is required for Laplacian encoding")
            return LaplacianEmbedding(self.mesh, self.cfg.encoding).to(self.device)

        elif self.cfg.encoding_type == "positional":
            return PositionalEmbedding(**self.cfg.encoding).to(self.device)

        else:
            raise ValueError(f"Unknown encoding type: {self.cfg.encoding_type}")

    def requires_face_ids(self) -> bool:
        return self.cfg.encoding_type == "laplacian"

    def _preprocess(self, pts: Float[Tensor, "P in_dim"]) -> Float[Tensor, "P in_dim"]:
        pts = (pts - self.scene_min) / (self.scene_max - self.scene_min)
        pts = 2 * pts + 1
        return pts

    def forward(
        self, pts: Float[Tensor, "P in_dim"], **kwargs
    ) -> Float[Tensor, "P out_dim"]:
        pts_norm = self._preprocess(pts)

        if self.cfg.encoding_type == "laplacian":
            net_in = self.encoding_model(pts, **kwargs)
        else:
            enc = self.encoding_model(pts_norm, **kwargs)
            if self.cfg.encoding_type == "hash":
                net_in = torch.cat((pts_norm, enc), dim=-1)
            else:
                net_in = enc
        out = self.network(net_in)
        return out
