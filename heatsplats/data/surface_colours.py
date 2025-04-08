from dataclasses import dataclass

import torch
import trimesh
import logging

from termcolor import colored

import heatsplats.utils as utils
import heatsplats
from heatsplats.utils import BaseObject
from heatsplats.utils.typing import *

__all__ = ["SurfaceColoursDataLoader"]


class BaseMeshColourDataset:
    def __init__(self, mesh: trimesh.Trimesh, device: str = "cpu"):
        self.device = device
        self._mesh = mesh
        self._verts = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
        self._faces = torch.tensor(mesh.faces, device=device)

    def get_vertex_colours(self):
        try:
            self._mesh.visual.vertex_colors
        except AttributeError:
            self._mesh.visual = self._mesh.visual.to_color()
        return torch.tensor(
            self._mesh.visual.vertex_colors[:, :3] / 255,
            dtype=torch.float,
            device=self.device,
        )

    def get_texture_image(self):
        try:
            tex_img = self._mesh.visual.material.baseColorTexture
            if tex_img is None:
                raise AttributeError
        except AttributeError:
            tex_img = self._mesh.visual.material.image
            if tex_img is None:
                raise AttributeError
        return tex_img

    @property
    def collate_fn(self):
        return self.collate if hasattr(self, "collate") else None


@heatsplats.register("all-vertex-colours-dataset")
class AllVertexColoursDataset(BaseMeshColourDataset, torch.utils.data.Dataset):
    def __init__(self, mesh: trimesh.Trimesh, device: str = "cpu"):
        super().__init__(mesh, device)
        self.vcols = self.get_vertex_colours()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"pos": self._verts, "colour": self.vcols}

    def collate(self, batch):
        return batch[0]


@heatsplats.register("vertex-colours-sampler-dataset")
class VertexColourSamplerDataset(BaseMeshColourDataset, torch.utils.data.Dataset):
    def __init__(self, mesh: trimesh.Trimesh, device: str = "cpu"):
        super().__init__(mesh, device)
        self.vcols = self.get_vertex_colours()

    def __len__(self) -> int:
        return self.vcols.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"pos": self._verts[idx], "colour": self.vcols[idx], "vert_id": idx}


@heatsplats.register("random-colours-sampler-dataset")
class RandomColourSamplerDataset(
    BaseMeshColourDataset, torch.utils.data.IterableDataset
):
    def __init__(self, mesh: trimesh.Trimesh, device: str = "cpu"):
        super().__init__(mesh, device)
        self._uv = torch.tensor(self._mesh.visual.uv)
        self._tex_img = self.get_texture_image()

    def collate(self, batch):
        batch_size = len(batch)
        face_id = torch.randint(
            0, self._faces.shape[0], (batch_size,), device=self.device
        )
        bary_coord = utils.uniform_sample_triangle(
            torch.rand((batch_size, 2), device=self.device)
        )
        pos = utils.interpolate_barycentric_coords(
            self._faces, face_id, bary_coord, self._verts
        )
        uv = utils.interpolate_barycentric_coords(
            self._faces, face_id, bary_coord, self._uv
        )
        color = trimesh.visual.uv_to_color(uv, self._tex_img)[:, :3] / 255
        color = torch.tensor(color, dtype=torch.float, device=self.device)
        batch = {"pos": pos, "colour": color, "face_id": face_id, "bary": bary_coord}
        return batch

    def __iter__(self):
        while True:
            yield {}


@heatsplats.register("known-heat-vertex-colours-dataset")
class KnownHeatVertexColoursDataset(BaseMeshColourDataset, torch.utils.data.Dataset):
    def __init__(self, mesh: trimesh.Trimesh, device: str = "cpu"):
        super().__init__(mesh, device)

    def __len__(self) -> int:
        return 1

    def configure(
        self,
        n_sources: int = 3,
        source_sampling_method: str = "fps",
        diff_time_scaler_func: Callable = lambda x: x,
    ):
        self.n_sources = n_sources

        if n_sources == 3:
            self._source_idxs = torch.tensor([3804, 0, 4274], device=self.device)

            self._gt_splats = {
                "angles": torch.tensor([45.0, 18.3, 10.0], device=self.device),
                "anisotropies": torch.tensor([33.0, 60, 5.2], device=self.device),
                "diff_times": torch.tensor([0.001, 0.1, 0.01], device=self.device),
                "kernel_colours": torch.tensor(
                    [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                    dtype=torch.float,
                    device=self.device,
                ),
            }
        else:
            if source_sampling_method == "fps":
                source_idx = utils.farthest_point_sampling(self._verts, n_sources)
                self._source_idxs = source_idx.nonzero(as_tuple=True)[0]
            else:
                self._source_idxs = torch.randint(
                    0, len(self._verts), (n_sources,), device=self.device
                )

            self._gt_splats = {
                "angles": torch.rand(n_sources, device=self.device) * 180,
                "anisotropies": (100 * torch.rand(n_sources, device=self.device)),
                "diff_times": diff_time_scaler_func(
                    torch.rand(n_sources, device=self.device)
                ),
                "kernel_colours": torch.rand(
                    (n_sources, 3),
                    dtype=torch.float,
                    device=self.device,
                ),
            }

    @property
    def gt_spalts(self):
        return self._gt_splats

    @property
    def source_idxs(self):
        return self._source_idxs

    def bake_heat(self, albo_evals, albo_evecs, mass, normalise_colours):
        idx_range = torch.arange(self.n_sources, device=self.device)

        gt_colours = torch.zeros(
            [self.n_sources, *self._verts.shape], device=self.device
        )
        gt_colours[idx_range, self._source_idxs, :] = self._gt_splats["kernel_colours"]

        gt_colours = utils.heat_diffusion(
            gt_colours.to(self.device),
            mass,
            albo_evals,
            albo_evecs,
            self._gt_splats["diff_times"],
        )

        gta = self._gt_splats["angles"].detach().cpu().numpy()
        gts = self._gt_splats["anisotropies"].detach().cpu().numpy()
        gtt = self._gt_splats["diff_times"].detach().cpu().numpy()
        gtc = self._gt_splats["kernel_colours"].detach().cpu().numpy()

        logger = logging.getLogger("heatsplats")
        logger.info(
            f"GT -> ",
            colored(f"Angles: {gta}, ", "yellow"),
            colored(f"Anisotropies: {gts}, ", "green"),
            colored(f"Diff times: {gtt}, ", "blue"),
            colored(f"Kernel colours: {gtc}", "red"),
        )

        gt_colours = gt_colours.sum(dim=0)
        if normalise_colours:
            gt_colours = utils.normalise_colours(gt_colours)

        self.vcols = gt_colours

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"pos": self._verts, "colour": self.vcols}

    def collate(self, batch):
        return batch[0]


class SurfaceColoursDataLoader(BaseObject):

    @dataclass
    class Config(BaseObject.Config):
        batch_size: int = 1
        num_workers: int = 4
        dataset_type: str = "all-vertex-colours-dataset"

    cfg: Config

    def configure(self, mesh: trimesh.Trimesh):
        self._mesh = mesh

    def get_loader(self) -> torch.utils.data.DataLoader:
        dataset = heatsplats.find(self.cfg.dataset_type)(self._mesh, self.device)
        collate_fn = dataset.collate_fn
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
            collate_fn=collate_fn,
        )
        return dataloader
