from dataclasses import dataclass, field

import torch

import hktex
from hktex.modules.base import TextureModel
from hktex.modules.tracer import GeodesicTracer
from hktex.utils.typing import *

from .pcl_texture import PCLTexture

__all__ = ["PCLModel"]


@hktex.register("modules.pcl-model")
class PCLModel(TextureModel):
    @dataclass
    class Config(TextureModel.Config):
        model_type: str = "modules.pcl-texture"
        model: dict = field(default_factory=dict)

        tracer_type: str = ""
        tracer: dict = field(default_factory=dict)

        point_batching: Optional[int] = None

    cfg: Config

    def configure(
        self,
        **kwargs,
    ):
        super().configure()

        self.mesh = kwargs["mesh"]
        assert self.mesh is not None

        TextureClass = hktex.find(self.cfg.model_type)
        self.model: PCLTexture = TextureClass(self.cfg.model, self.mesh)

        self.tracer: Optional[GeodesicTracer] = None
        if self.cfg.tracer_type:
            self.tracer = hktex.find(self.cfg.tracer_type)(self.cfg.tracer, self.mesh)
        self._inference_mode = False

    def inference_mode(self, new_mode: bool) -> bool:
        old_mode = self._inference_mode
        self._inference_mode = new_mode
        return old_mode

    def requires_scene_bounds(self):
        return False

    def requires_face_ids(self):
        return True

    def post_optimizer_step(self):
        self.model.post_optimizer_step()

    def mark_knn_dirty(self):
        if hasattr(self.model, "mark_knn_dirty"):
            self.model.mark_knn_dirty()

    def prepare_kernels(self, save_barycentric: bool = True):
        return self.model.prepare_kernels(self.mesh, save_barycentric=save_barycentric)

    def prepare_points(
        self,
        pts: Float[Tensor, "P in_dim"] | None,
        face_ids: Int[Tensor, "P"],
        barys: Float[Tensor, "P 3"] | None = None,
    ):
        return self.model.prepare_points(
            mesh=self.mesh,
            face_ids=face_ids,
            barys=barys,
            pts=pts,
        )

    def _forward_batch(
        self,
        pts_batch: Float[Tensor, "p in_dim"] | None,
        face_ids_batch: Int[Tensor, "p"],
        barys_batch: Float[Tensor, "p 3"] | None,
    ):
        points_info = self.prepare_points(
            pts=pts_batch,
            face_ids=face_ids_batch,
            barys=barys_batch,
        )
        colours_batch, _, _, _ = self.model.interpolate_colours(points_info)
        colours_batch = self.model(colours_batch)
        return colours_batch

    def forward(
        self, pts: Float[Tensor, "P in_dim"], **kwargs
    ) -> Float[Tensor, "P out_dim"]:
        face_ids: Tensor = kwargs["face_ids"].to(torch.int)
        barys: Optional[Tensor] = kwargs.get("barys", None)

        P = pts.shape[0]
        assert P == face_ids.shape[0]
        if barys is not None:
            assert barys.shape[0] == P

        # Match the heat-kernel KNN lifecycle: build/query graph each forward unless in inference mode.
        if not self._inference_mode:
            self.prepare_kernels(save_barycentric=True)

        batch_size = self.cfg.point_batching
        if batch_size is None:
            return self._forward_batch(pts, face_ids, barys)

        colours = []
        for i in range(0, P, batch_size):
            pts_batch = pts[i : i + batch_size]
            face_ids_batch = face_ids[i : i + batch_size]
            barys_batch = None if barys is None else barys[i : i + batch_size]
            colours_batch = self._forward_batch(pts_batch, face_ids_batch, barys_batch)
            colours.append(colours_batch)
        return torch.cat(colours, dim=0)

    def save_torch(self, filename):
        self.model.save_torch(filename)

    def save_numpy_npz(self, filename):
        self.model.save_numpy_npz(filename)

    def load_torch(self, filename):
        self.model.load_torch(filename)
