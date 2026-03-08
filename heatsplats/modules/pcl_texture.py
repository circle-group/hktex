from dataclasses import dataclass, field
import numpy as np
from termcolor import colored

import torch
import torch.nn as nn

import heatsplats
import heatsplats.utils as utils
from heatsplats.utils import BaseModule
from heatsplats.utils.typing import *
import heatsplats.knn_heat as knn_heat

from .mesh import Mesh

__all__ = ["PCLTexture"]


@heatsplats.register("modules.pcl-texture")
class PCLTexture(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        n_sources: int = 1024
        out_dim: int = 3

        feature_dim: int = 16
        out_net: bool = False
        normalize_colours: bool = False

        allow_negative_colours: bool = False
        range_enforcement_type: str = "activations"  # "pgd" | "activations"

        knn_k: int = 8
        weighting: str = "softmax_rbf"  # "softmax_rbf" | "inverse_distance"
        softmax_temperature: float = 0.05
        distance_eps: float = 1e-8

        # Optional normal-aware gate using existing mesh normals.
        use_face_normal_weighting: bool = False
        normal_weight_beta: float = 8.0

        faiss: knn_heat.FaissGpuIndexConfig = field(
            default_factory=knn_heat.FaissGpuIndexConfig
        )

    cfg: Config

    _mean_colour: Float[Tensor, "1 D"]
    _kernel_colours: Float[Tensor, "G D_or_F"]
    _kernel_locations: Float[Tensor, "G 3"]
    _kernel_face_ids: Int[Tensor, "G"]
    _softmax_temperature_raw: Float[Tensor, "1"]

    def configure(
        self,
        mesh: Mesh,
        **kwargs,
    ):
        super().configure()
        self.__mesh = mesh

        self.N_sources = self.cfg.n_sources
        self.out_dim = self.cfg.out_dim
        self.normalize_colours = self.cfg.normalize_colours

        if self.cfg.range_enforcement_type == "activations":
            if self.cfg.allow_negative_colours:
                self._colour_act = lambda x: torch.tanh(x)
                self._inv_colour_act = lambda x: torch.atanh(
                    x.clamp(-0.999999, 0.999999)
                )
            else:
                self._colour_act = lambda x: torch.sigmoid(x)
                self._inv_colour_act = lambda x: torch.logit(x.clamp(1e-6, 1.0 - 1e-6))
            self._tau_act = lambda x: torch.nn.functional.softplus(x)
            self._inv_tau_act = lambda x: torch.log(torch.expm1(x))
        elif self.cfg.range_enforcement_type == "pgd":
            self._colour_act = lambda x: x
            self._inv_colour_act = lambda x: x
            self._tau_act = lambda x: x
            self._inv_tau_act = lambda x: x
        else:
            raise ValueError(
                f"Unknown range enforcement type: {self.cfg.range_enforcement_type}"
            )

        self.out_net = None
        if self.cfg.out_net:
            fdim = self.cfg.feature_dim
            self.out_net = nn.Sequential(
                nn.ReLU(),
                nn.Linear(fdim, 2 * fdim),
                nn.ReLU(),
                nn.Linear(2 * fdim, self.out_dim),
                nn.Sigmoid(),
            ).to(self.device)

        self._make_splats(mesh)

        self._index_dirty = True
        self._faiss_index = knn_heat.FaissGpuFlatIndex(self.cfg.faiss)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        nb = super().named_buffers(prefix, recurse, remove_duplicate)
        for name, buf in nb:
            if "__mesh" not in name:
                yield name, buf

    def _make_splats(self, mesh: Mesh):
        factory_kwargs = {"dtype": torch.float, "device": self.device}
        colour_dim = self.cfg.feature_dim if self.out_net is not None else self.out_dim
        kernel_colours = torch.rand((self.N_sources, colour_dim), **factory_kwargs)
        if self.cfg.allow_negative_colours:
            kernel_colours = kernel_colours * 2.0 - 1.0

        fids, bary = utils.uniform_sampling(
            mesh.verts, mesh.faces, max(3 * self.N_sources, 10_000)
        )
        pos = mesh.barycentric_to_cartesian(bary, mesh.get_face_vertices(fids))
        mask = utils.farthest_point_sampling(pos, self.N_sources)
        kernel_locations = pos[mask]
        kernel_face_ids = fids[mask]

        mean_colour = torch.mean(kernel_colours, dim=0, keepdim=True)
        residual_colours = kernel_colours - mean_colour

        self._mean_colour = torch.nn.Parameter(mean_colour)
        self._kernel_colours = torch.nn.Parameter(residual_colours)
        self._kernel_locations = torch.nn.Parameter(kernel_locations)
        self._kernel_face_ids = nn.Buffer(kernel_face_ids, persistent=True)
        init_tau = torch.tensor(
            [max(float(self.cfg.softmax_temperature), 1e-8)],
            dtype=torch.float,
            device=self.device,
        )
        self._softmax_temperature_raw = torch.nn.Parameter(self._inv_tau_act(init_tau))

        self.splat_param_keys = ["kernel_colours", "softmax_temperature"]

    @property
    def kernel_colours(self) -> Float[Tensor, "G D"]:
        return self._colour_act(self._kernel_colours)

    @property
    def kernel_locations(self) -> Float[Tensor, "G 3"]:
        return self._kernel_locations

    @property
    def kernel_face_ids(self) -> Int[Tensor, "G"]:
        return self._kernel_face_ids

    @property
    def softmax_temperature(self) -> Float[Tensor, "1"]:
        return self._tau_act(self._softmax_temperature_raw)

    def post_optimizer_step(self):
        if self.cfg.range_enforcement_type == "pgd":
            self.clamp_parameters()

    def mark_knn_dirty(self):
        self._index_dirty = True

    @torch.no_grad()
    def clamp_parameters(self):
        if self.cfg.allow_negative_colours:
            self._kernel_colours.clamp_(min=-1.0, max=1.0)
        else:
            self._kernel_colours.clamp_(min=0.0, max=1.0)
        # Mean colour is always a non-negative base tone.
        self._mean_colour.clamp_(min=0.0, max=1.0)
        self._softmax_temperature_raw.clamp_(min=1e-8)

    def prepare_kernels(
        self,
        mesh: Mesh | None = None,
        save_barycentric: bool = True,
        **kwargs,
    ) -> dict[str, Tensor]:
        mesh = mesh if mesh is not None else self.__mesh
        kernel_vert_idx = mesh.get_face_vertices(self.kernel_face_ids)
        kernel_barycentric_coords = mesh.cartesian_to_barycentric(
            self.kernel_locations, kernel_vert_idx
        )

        if save_barycentric:
            self.save_barycentric_locations(kernel_barycentric_coords)

        if self._index_dirty:
            # add 0.0 to make the parameter to a tensor
            # since faiss incorrectly checks the exact type as assert type(x) is torch.Tensor
            self._faiss_index.build((self.kernel_locations + 0.0).contiguous())
            self._index_dirty = False

        return {
            "vert_idx": kernel_vert_idx,
            "barycentric_coords": kernel_barycentric_coords,
        }

    def prepare_points(
        self,
        mesh: Mesh | None = None,
        face_ids: Int[Tensor, "P"] = None,
        barys: Float[Tensor, "P 3"] | None = None,
        pts: Float[Tensor, "P 3"] | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        mesh = mesh if mesh is not None else self.__mesh
        if face_ids is None:
            raise ValueError("face_ids must be provided to prepare_points.")

        if barys is None and pts is not None:
            pass
        elif barys is not None and pts is None:
            pts_tri_vert_idx = mesh.get_face_vertices(face_ids.to(torch.int))
            pts = mesh.barycentric_to_cartesian(barys, pts_tri_vert_idx)
        else:
            raise ValueError("Either barys or pts must be provided to prepare_points.")

        if self._index_dirty:
            raise RuntimeError(
                "KNN index is stale/unbuilt. Call prepare_kernels() before prepare_points()."
            )

        k = int(max(1, min(self.cfg.knn_k, self.N_sources)))
        _, nn_indices = self._faiss_index.search(pts.contiguous(), k=k)
        nn_dists = self._faiss_index.knn_distances(pts.contiguous(), nn_indices)

        return {
            "pts": pts,
            "face_ids": face_ids.to(torch.int),
            "nn_indices": nn_indices.to(torch.int64),
            "nn_dists": nn_dists,
        }

    def _compute_weights(
        self,
        dists: Float[Tensor, "P k"],
        query_face_ids: Int[Tensor, "P"],
        nn_indices: Int[Tensor, "P k"],
    ) -> Float[Tensor, "P k"]:
        # FaissGpuFlatIndex with metric="l2" returns squared L2 distances.
        d2 = dists
        eps = self.cfg.distance_eps
        if self.cfg.weighting == "softmax_rbf":
            tau2 = self.softmax_temperature.clamp_min(1e-8).pow(2)
            logits = -d2 / tau2
            weights = torch.softmax(logits, dim=1)
        elif self.cfg.weighting == "inverse_distance":
            # Convert squared distance to Euclidean distance for inverse weighting.
            inv = torch.rsqrt(d2 + eps)
            weights = inv / (inv.sum(dim=1, keepdim=True) + eps)
        else:
            raise ValueError(f"Unknown weighting: {self.cfg.weighting}")

        if self.cfg.use_face_normal_weighting:
            qn = self.__mesh.fnorms[query_face_ids]  # [P, 3]
            src_n = self.__mesh.fnorms[self.kernel_face_ids[nn_indices]]  # [P, k, 3]
            dots = (src_n * qn.unsqueeze(1)).sum(dim=-1).clamp(min=-1.0, max=1.0)
            gate = torch.exp(self.cfg.normal_weight_beta * (dots - 1.0))
            weights = weights * gate
            weights = weights / (weights.sum(dim=1, keepdim=True) + eps)

        return weights

    def interpolate_colours(
        self,
        pts_info: dict[str, Tensor],
    ) -> tuple[
        Float[Tensor, "P D"], None, Int[Tensor, "k P 1"], Float[Tensor, "k P 1"]
    ]:
        face_ids: Int[Tensor, "P"] = pts_info["face_ids"]
        nn_indices: Int[Tensor, "P k"] = pts_info["nn_indices"]
        dists: Float[Tensor, "P k"] = pts_info["nn_dists"]

        weights = self._compute_weights(dists, face_ids, nn_indices)
        kernel_colours = self.kernel_colours[nn_indices]  # [P, k, D]
        colours = (weights.unsqueeze(-1) * kernel_colours).sum(dim=1)
        colours = (self._mean_colour + colours).clamp(min=0.0, max=1.0)

        topk_kernel_idxs = nn_indices.transpose(0, 1).unsqueeze(-1)
        topk_kernel_contribs = weights.transpose(0, 1).unsqueeze(-1)
        return colours, None, topk_kernel_idxs, topk_kernel_contribs

    # Compatibility adapter to match the HeatKernelTextureKNN call-site shape.
    def diffuse_heat_kernels(self, pts_info, **kwargs):
        return self.interpolate_colours(pts_info=pts_info)

    # Compatibility no-op for KNN trainer flow.
    def reset(self, **kwargs):
        return None

    def forward(self, x_interp: Float[Tensor, "P D"]) -> Float[Tensor, "P out_dim"]:
        out = x_interp
        if self.out_net is not None:
            out = self.out_net(out)
        if self.normalize_colours:
            out = utils.normalise_colours(out)
        return out

    def compute_vertex_colours(self, mesh: Mesh):
        # Build one incident face + one-hot barycentric coordinates per vertex.
        V = mesh.N_verts
        F = mesh.N_faces
        face_ids = torch.full((V,), F, dtype=torch.long, device=self.device)
        face_idx = torch.arange(F, device=self.device, dtype=torch.long)
        faces = mesh.faces.long()

        for local_vid in range(3):
            face_ids.scatter_reduce_(
                0,
                faces[:, local_vid],
                face_idx,
                reduce="amin",
                include_self=True,
            )

        if torch.any(face_ids == F):
            raise RuntimeError("Found vertex with no incident face.")

        chosen_faces = faces[face_ids]  # [V, 3]
        v_ids = torch.arange(V, device=self.device).unsqueeze(1)
        match = chosen_faces.eq(v_ids)
        local_idx = match.float().argmax(dim=1)

        bary = torch.zeros((V, 3), device=self.device)
        bary[torch.arange(V, device=self.device), local_idx] = 1.0

        self.prepare_kernels(mesh, save_barycentric=False)
        pts_info = self.prepare_points(
            mesh=mesh,
            face_ids=face_ids.to(torch.int),
            barys=bary,
            pts=None,
        )
        colours, _, _, _ = self.interpolate_colours(pts_info)
        return self.forward(colours)

    @property
    def colored_print_opt_params(self):
        kernel_colours = self.kernel_colours.view(-1).detach().cpu().numpy()
        return colored(f"Kernel colours: {kernel_colours}", "red")

    def save_barycentric_locations(self, barycentric_coords):
        setattr(
            self._kernel_locations,
            "bary_coords",
            barycentric_coords.detach(),
        )

    def save_torch(self, filename, compressed: bool = True):
        if compressed:
            state_dict = {
                k: v.half() if v.is_floating_point() else v
                for k, v in self.state_dict().items()
            }
        else:
            state_dict = self.state_dict()
        torch.save(state_dict, filename)

    def save_numpy_npz(self, filename, compressed: bool = True):
        np_dict = {}
        for k, v in self.state_dict().items():
            if "__mesh" in k:
                continue
            if compressed and v.is_floating_point():
                np_dict[k] = v.half().detach().cpu().numpy()
            else:
                np_dict[k] = v.detach().cpu().numpy()
        np.savez_compressed(filename, **np_dict)

    def load_torch(self, filename: str):
        sd = torch.load(filename, map_location=self.device, weights_only=True)
        self.load_state_dict(sd, strict=False)
        self.float()
        self._index_dirty = True

    def load_numpy_npz(self, filename: str):
        if filename.endswith(".pt"):
            filename = filename.replace(".pt", ".npz")
        np_dict = np.load(filename, allow_pickle=False)
        sd = {k: torch.tensor(v, device=self.device) for k, v in np_dict.items()}
        self.load_state_dict(sd, strict=False)
        self.float()
        self._index_dirty = True
