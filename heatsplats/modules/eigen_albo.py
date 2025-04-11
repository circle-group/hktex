from dataclasses import dataclass, field
import math

import torch
import torch.linalg as linalg

from tqdm import tqdm

import heatsplats
from heatsplats.utils import get_anisotropic_lbo, compute_eig_laplacian, BaseObject
from heatsplats.utils.typing import *

from .mesh import Mesh

__all__ = ["EigenAlboInterpolation"]


@heatsplats.register("modules.eigen-albo-interpolation")
class EigenAlboInterpolation(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        k_eig: int = 256

        use_precomputed: bool = True
        precompute_anisotropies: list = field(
            default_factory=lambda: [1, 2.5, 5, 7.5, 10, 25, 50, 75, 100]
        )
        precompute_angles_every_deg: int = 30
        mesh_path: Optional[str] = None
        precomputed_name: str = "eigen_albo"

    cfg: Config

    def configure(self, mesh: Mesh):
        self._mesh = mesh

        _all_eigen, _smp_coords, _mass = self.precompute_all_eigen()

        M = _all_eigen.shape[0]

        self._mass = _mass
        self._smp_coords_cartesian = torch.stack(
            [
                torch.cos(_smp_coords[:, 0]) * _smp_coords[:, 1],
                torch.sin(_smp_coords[:, 0]) * _smp_coords[:, 1],
            ],
            dim=1,
        )

        k_eig = self.cfg.k_eig
        self._eigen_val = _all_eigen[:, :k_eig].contiguous()
        self._eigen_vec = _all_eigen[:, k_eig:].view(M, -1, k_eig).contiguous()

        self.M = M
        self.K = k_eig

        V = self._mesh.N_verts

        assert self._smp_coords_cartesian.shape[0] == M
        assert self._eigen_vec.shape[1] == V and self._mass.shape[1] == V

    def precompute_all_eigen(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fpath = self.cfg.mesh_path
        if self.cfg.use_precomputed and fpath is None:
            heatsplats.warn(
                f"Eigen Albo Interpolation requires mesh path when using precomputed, falling back to non-precomputed"
            )

        # Essentially just a wrapper for _precompute_all_eigen which makes sure
        # that the precomputed values are saved and loaded if possible
        if fpath is None or not self.cfg.use_precomputed:
            all_eigen, sampling_coords, mass = self._precompute_all_eigen()
        else:
            fpath_base = fpath.rsplit(".", 1)[0]
            precomputed_path = f"{fpath_base}_{self.cfg.precomputed_name}.pt"
            heatsplats.info(f"Loading precomputed albo eigen from {precomputed_path}")
            try:
                precomputed = torch.load(precomputed_path, weights_only=True)
                all_eigen = precomputed["all_eigen"]
                sampling_coords = precomputed["sampling_coords"]
                mass = precomputed["mass"]
            except (FileNotFoundError, KeyError):
                heatsplats.info(f"Precomputed albo eigen not found")
                all_eigen, sampling_coords, mass = self._precompute_all_eigen()
                torch.save(
                    {
                        "all_eigen": all_eigen,
                        "sampling_coords": sampling_coords,
                        "mass": mass,
                    },
                    precomputed_path,
                )
        return (
            all_eigen.to(torch.float32).to(self.device),
            sampling_coords.to(torch.float32).to(self.device),
            mass.to(torch.float32).to(self.device),
        )

    def _precompute_all_eigen(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_eigen = []
        sampling_coords = []

        # Compute eigenvalues and eigenvectors obtained eigendecomposing
        # the Anisotropic Laplacian for different rotations and anisotropies
        heatsplats.info("Computing all eigendecompositions")
        for angle in tqdm(range(0, 180, self.cfg.precompute_angles_every_deg)):
            angle = math.radians(angle)
            for scale in self.cfg.precompute_anisotropies:
                sampling_coords.append(torch.tensor([angle, scale]))

                lapl, mass = get_anisotropic_lbo(
                    self._mesh.verts,
                    self._mesh.faces.T,
                    self._mesh.fnorms,
                    rotation_angle=angle,
                    anisotropy=float(scale),
                )

                eval, evecs = compute_eig_laplacian(lapl, mass, self.cfg.k_eig)

                flat_evecs = torch.tensor(evecs).flatten()
                all_eigen.append(torch.cat([torch.tensor(eval), flat_evecs]))

        mass = torch.tensor(mass).unsqueeze(0).contiguous()

        polar_smp_coords = torch.stack(sampling_coords)

        return torch.stack(all_eigen), polar_smp_coords, mass

    def interpolate_anisotropies(
        self,
        angles: Float[Tensor, "G"],
        scales: Float[Tensor, "G"],
    ) -> Float[Tensor, "G M"]:
        G = angles.shape[0]
        assert scales.shape[0] == G

        query_cartesian = torch.stack(
            [
                torch.cos(angles) * scales,
                torch.sin(angles) * scales,
            ],
            dim=1,
        )

        diff = self._smp_coords_cartesian.unsqueeze(1) - query_cartesian.unsqueeze(0)
        squared_distance = (diff * diff).sum(-1, keepdim=True)
        dist, idx = squared_distance.topk(k=4, largest=False, dim=0)
        x_idx = idx.squeeze().t()  # B, 4
        dist = dist.squeeze().t()  # B, 4

        weights = 1.0 / torch.clamp(dist, min=1e-16)
        weights = weights / weights.sum(dim=1, keepdim=True)

        M = self._eigen_val.shape[0]
        albo_weights = weights.new_zeros((G, M)).scatter_(1, index=x_idx, src=weights)

        return albo_weights

    def albo_vertices(
        self,
        albo_weights: Float[Tensor, "G M"],
        vert_idx: Optional[Int[Tensor, "P"]] = None,
    ) -> Tuple[Float[Tensor, "G K"], Float[Tensor, "G P K"], Float[Tensor, "G P"]]:
        G, M = albo_weights.shape[0], self._eigen_val.shape[0]
        assert albo_weights.shape[1] == M

        with torch.no_grad():
            eigen_vec, mass = self._eigen_vec, self._mass
            M, _, K = eigen_vec.shape

            if vert_idx is not None:
                eigen_vec = eigen_vec[:, vert_idx]
                mass = mass[:, vert_idx]

        evals = albo_weights @ self._eigen_val
        evecs = torch.einsum("ij,jkl->ikl", albo_weights, eigen_vec)

        return evals, evecs, mass

    def barycentric_albo_points(
        self,
        albo_weights: Float[Tensor, "G M"],
        barycentric_coords: Float[Tensor, "P 3"],
        vert_idx: Int[Tensor, "P 3"],
    ) -> Tuple[Float[Tensor, "G K"], Float[Tensor, "G P K"], Float[Tensor, "G P"]]:
        G, P = albo_weights.shape[0], barycentric_coords.shape[0]
        M = self._eigen_val.shape[0]
        assert vert_idx.shape[0] == P and albo_weights.shape[1] == M
        assert barycentric_coords.shape[1] == 3 and vert_idx.shape[1] == 3

        with torch.no_grad():
            eigen_vec, mass = self._eigen_vec, self._mass
            M, _, K = eigen_vec.shape

            eigen_vec = eigen_vec[:, vert_idx.view(-1)]  # M, P*3, K
            mass = mass[:, vert_idx.view(-1)].view(1, P, 3)  # 1, P, 3

        # G,M x M,K -> G,K
        evals = albo_weights @ self._eigen_val
        # G,M x M,P*3,K -> G,P*3,K -> G, P, 3, K
        evecs = torch.einsum("ij,jkl->ikl", albo_weights, eigen_vec).view(G, P, 3, K)

        bary_W = barycentric_coords.unsqueeze(0)  # 1, P, 3

        # 1,P,1,3 x B,P,3,K -> B,P,K
        evec_interp = torch.matmul(bary_W.unsqueeze(2), evecs).squeeze(2)
        mass_interp = linalg.vecdot(bary_W, mass)  # 1, P

        return evals, evec_interp, mass_interp

    def barycentric_albo_gaussians(
        self,
        albo_weights: Float[Tensor, "G M"],
        barycentric_coords: Float[Tensor, "G 3"],
        vert_idx: Int[Tensor, "G 3"],
    ) -> Tuple[Float[Tensor, "G K"], Float[Tensor, "G"]]:
        G, M = albo_weights.shape[0], self._eigen_val.shape[0]
        assert barycentric_coords.shape[0] == G and vert_idx.shape[0] == G
        assert albo_weights.shape[1] == M
        assert barycentric_coords.shape[1] == 3 and vert_idx.shape[1] == 3

        with torch.no_grad():
            eigen_vec, mass = self._eigen_vec, self._mass
            M, _, K = eigen_vec.shape

            eigen_vec = eigen_vec[:, vert_idx.view(-1)].view(M, G, 3, K)  # M, G, 3, K
            eigen_vec = eigen_vec.permute(1, 0, 2, 3)  # G, M, 3, K
            mass = mass[:, vert_idx.view(-1)].view(G, 3)  # G, 3

        # G,M x G,M,3,K -> G,3,K
        evecs = linalg.vecdot(
            albo_weights.unsqueeze(-1).unsqueeze(-1), eigen_vec, dim=1
        )

        bary_W = barycentric_coords  # B, 3

        evec_interp = torch.einsum("bt,btk->bk", bary_W, evecs)
        mass_interp = linalg.vecdot(bary_W, mass)

        return evec_interp, mass_interp
