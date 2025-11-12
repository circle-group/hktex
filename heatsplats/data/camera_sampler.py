from dataclasses import dataclass, field
import trimesh
from omegaconf import OmegaConf

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

T = TypeVar("T")
BoundsType = tuple[T, T]


@dataclass
class BoundsConfig:
    camera_distance: Optional[BoundsType[float]] = None
    azimuth_deg: Optional[BoundsType[float]] = None
    elevation_deg: Optional[BoundsType[float]] = None

    img_width: Optional[BoundsType[int]] = None
    img_height: Optional[BoundsType[int]] = None

    fov: Optional[BoundsType[float]] = None


@dataclass
class CameraSamplerDataConfig(MeshSamplerDataConfig):
    batch_size: int = 4
    num_workers: int = 4

    bounds: BoundsConfig = field(default_factory=BoundsConfig)

    tile_width: Optional[List[int]] = None  # 1 for fixed, 2 for sampling in bounds
    tile_height: Optional[List[int]] = None

    img_width: Optional[int] = None
    img_height: Optional[int] = None


class CameraSamplerDataset(IterableDataset):
    def __init__(self, cfg: CameraSamplerDataConfig, mesh: trimesh.Trimesh, split: str):
        super().__init__()

        self.cfg = cfg
        self.mesh = mesh
        self.split = split

        bounds = OmegaConf.to_container(self.cfg.bounds, resolve=True)
        self.bounds = {k: v for k, v in bounds.items() if v is not None}

        self.tile_w, self.tile_h = self.cfg.tile_width, self.cfg.tile_height
        if self.tile_w is not None or self.tile_h is not None:
            if self.tile_w is None or self.tile_h is None:
                raise ValueError(
                    "Setting tile size requires setting both tile width and tile height"
                )

            if self.cfg.img_width is None and self.cfg.bounds.img_width is None:
                raise ValueError(
                    "Setting tile size requires either the img_width or img_width bounds"
                )

            if self.cfg.img_height is None and self.cfg.bounds.img_height is None:
                raise ValueError(
                    "Setting tile size requires either the img_height or img_height bounds"
                )
            if len(self.tile_w) == 1:
                self.tile_w = self.tile_w[0]
            if len(self.tile_h) == 1:
                self.tile_h = self.tile_h[0]

    def sample_field(self, bounds, N: int = 1):
        # no pure tuple support in omegaconf
        if (isinstance(bounds, tuple) or isinstance(bounds, list)) and len(bounds) == 2:
            low, high = bounds[0], bounds[1]
            if isinstance(low, float) or isinstance(high, float):
                return torch.empty((N,), dtype=torch.float32).uniform_(low, high)
            if isinstance(low, int) and isinstance(high, int):
                return torch.empty((N,), dtype=torch.int32).random_(low, high + 1)
        raise ValueError(f"Unknown bounds object {bounds}")

    def has_tiles(self):
        return self.tile_w is not None

    def sample_tiles(self, cameras: dict, N: int = 1):
        assert self.has_tiles()

        if isinstance(self.tile_w, int):
            tile_w = torch.full((N,), self.tile_w, dtype=torch.int32)
        else:
            tile_w = torch.empty((N,), dtype=torch.int32).random_(
                self.tile_w[0], self.tile_w[1] + 1
            )

        if isinstance(self.tile_h, int):
            tile_h = torch.full((N,), self.tile_h, dtype=torch.int32)
        else:
            tile_h = torch.empty((N,), dtype=torch.int32).random_(
                self.tile_h[0], self.tile_h[1] + 1
            )

        if "img_width" in cameras:
            img_width = cameras["img_width"]
        else:
            img_width = torch.full((N,), self.cfg.img_width, dtype=torch.int32)

        if "img_height" in cameras:
            img_height = cameras["img_height"]
        else:
            img_height = torch.full((N,), self.cfg.img_height, dtype=torch.int32)

        # x_offset = torch.empty((N,), dtype=torch.int32).random_(0, img_width - tile_w)
        x_offset = torch.cat(
            [torch.randint(0, high, (1,)) for high in img_width - tile_w]
        )
        # y_offset = torch.empty((N,), dtype=torch.int32).random_(0, img_height - tile_h)
        y_offset = torch.cat(
            [torch.randint(0, high, (1,)) for high in img_height - tile_h]
        )

        return dict(
            crop_offset_x=x_offset,
            crop_offset_y=y_offset,
            crop_width=tile_w,
            crop_height=tile_h,
        )

    def sample_N(self, batch_size):
        cameras = dict()
        for field, bounds in self.bounds.items():
            cameras[field] = self.sample_field(bounds, batch_size)

        if self.has_tiles():
            tiles = self.sample_tiles(cameras, batch_size)
            cameras.update(tiles)

        return {"batch_size": batch_size, "cameras": cameras}

    def __iter__(self):
        while True:
            yield self.sample_N(self.cfg.batch_size)


@heatsplats.register("data.camera-sampler")
class CameraSamplerDataModule(MeshSamplerDataModule):
    cfg: CameraSamplerDataConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        cfg = parse_structured(CameraSamplerDataConfig, cfg)

        super().__init__(cfg=cfg)

    def setup(self, stage=None) -> None:
        super().setup(stage)
        if stage in [None, "fit"]:
            self.train_dataset = CameraSamplerDataset(self.cfg, self.mesh, "train")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=None,  # Disable automatic batching
            num_workers=self.cfg.num_workers,
        )
