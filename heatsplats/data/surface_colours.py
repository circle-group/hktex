from dataclasses import dataclass

import torch
import trimesh

import utils
import heatsplats
from heatsplats.utils import BaseObject
from heatsplats.utils.typing import *

__all__ = ["SurfaceColoursDataLoader"]


class BaseMeshColourDataset:
    def __init__(self, mesh: trimesh.Trimesh):
        super().__init__()
        self._mesh = mesh
        self._verts = torch.tensor(mesh.vertices, dtype=torch.float)
        self._faces = torch.tensor(mesh.faces)

    def get_vertex_colours(self):
        try:
            self._mesh.visual.vertex_colors
        except AttributeError:
            self._mesh.visual = self._mesh.visual.to_color()
        return torch.tensor(self._mesh.visual.vertex_colors[:, :3] / 255)

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
    def __init__(self, mesh: trimesh.Trimesh):
        super().__init__(mesh)
        self.vcols = self.get_vertex_colours()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"pos": self._verts, "colour": self.vcols}

    def collate(self, batch):
        return batch[0]


@heatsplats.register("vertex-colours-sampler-dataset")
class VertexColourSamplerDataset(BaseMeshColourDataset, torch.utils.data.Dataset):
    def __init__(self, mesh: trimesh.Trimesh):
        super().__init__(mesh)
        self.vcols = self.get_vertex_colours()

    def __len__(self) -> int:
        return self.vcols.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"pos": self._verts[idx], "colour": self.vcols[idx]}


@heatsplats.register("random-colours-sampler-dataset")
class RandomColourSamplerDataset(
    BaseMeshColourDataset, torch.utils.data.IterableDataset
):
    def __init__(self, mesh):
        super().__init__(mesh)
        self._uv = torch.tensor(self._mesh.visual.uv)
        self._tex_img = self.get_texture_image()

    def collate(self, batch):
        batch_size = len(batch)
        face_id = torch.randint(0, self._faces.shape[0], (batch_size,))
        bary_coord = utils.uniform_sample_triangle(torch.rand((batch_size, 2)))
        pos = utils.interpolate_barycentric_coords(
            self._faces, face_id, bary_coord, self._verts
        )
        uv = utils.interpolate_barycentric_coords(
            self._faces, face_id, bary_coord, self._uv
        )
        color = trimesh.visual.uv_to_color(uv, self._tex_img)[:, :3] / 255
        batch = {"pos": pos, "colour": color, "face_id": face_id, "bary": bary_coord}
        return batch

    def __iter__(self):
        while True:
            yield {}


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
        dataset = heatsplats.get(self.cfg.dataset_type)(self._mesh)
        collate_fn = dataset.collate_fn
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
            collate_fn=collate_fn,
        )
        return dataloader


if __name__ == "__main__":
    mesh = utils.load_mesh("../objects/spot/spot_triangulated.obj", merge_tex=False)

    dataset = AllVertexColoursDataset(mesh)
    collate_fn = dataset.collate_fn
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
    )

    for i, batch in enumerate(dataloader):
        print(i, batch["pos"].shape, batch["colour"].shape)
        if i > 10:
            break
