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

    def get_albo_eigenquantities(
        self,
        angles: Float[Tensor, "B"],
        scales: Float[Tensor, "B"],
        vert_idx: Optional[Int[Tensor, "D"]] = None,
    ) -> Tuple[Float[Tensor, "B K"], Float[Tensor, "B V K"], Float[Tensor, "B V"]]:
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

        eigen_vec, mass = self._eigen_vec, self._mass
        if vert_idx is not None:
            eigen_vec = eigen_vec[:, vert_idx]
            mass = mass[:, vert_idx]

        weights2 = weights.new_zeros(
            (weights.shape[0], self._eigen_val.shape[0])
        ).scatter_(1, index=x_idx, src=weights)
        evals = weights2 @ self._eigen_val
        evecs = torch.einsum("ij,jkl->ikl", weights2, eigen_vec)

        return evals, evecs, mass

    def barycentric_eig_interpolation(
        self,
        eigen_vec: Float[Tensor, "B V K"],
        mass: Float[Tensor, "1 V"],
        barycentric_coords: Float[Tensor, "B 3"],
        vert_idx: Int[Tensor, "B 3"],
    ) -> tuple[Float[Tensor, "B K"], Float[Tensor, "B"]]:

        target_eigen_vec = torch.take_along_dim(
            eigen_vec, vert_idx.unsqueeze(-1), dim=1
        )  # B 3 K
        target_mass = torch.take_along_dim(mass, vert_idx, dim=1)  # B 3

        W = barycentric_coords

        eigen_vec_interp = torch.einsum("bt,btk->bk", W, target_eigen_vec)  # B K
        mass_interp = linalg.vecdot(W, target_mass)  # B

        return eigen_vec_interp, mass_interp

    def barycentric_eig_interpolation_pts(
        self,
        eigen_vec: Float[Tensor, "B V K"],
        mass: Float[Tensor, "1 V"],
        barycentric_coords: Float[Tensor, "P 3"],
        vert_idx: Int[Tensor, "P 3"],
    ) -> tuple[Float[Tensor, "B P K"], Float[Tensor, "P"]]:

        B, V, K = eigen_vec.shape
        P = vert_idx.shape[0]

        # Gather eigen_vec values directly
        target_eigen_vec = eigen_vec[:, vert_idx.view(-1)].view(B, P, 3, K)  # B P 3 K

        # Gather mass values directly
        target_mass = mass[:, vert_idx.view(-1)].view(P, 3)  # P 3

        W = barycentric_coords  # P 3

        eigen_vec_interp = torch.matmul(W.unsqueeze(1), target_eigen_vec).squeeze(
            2
        )  # B P K
        mass_interp = linalg.vecdot(W, target_mass)  # P

        return eigen_vec_interp, mass_interp

    def barycentric_albo_eigenquantities(
        self,
        angles: Float[Tensor, "B"],
        scales: Float[Tensor, "B"],
        barycentric_coords: Float[Tensor, "P 3"],
        vert_idx: Int[Tensor, "P 3"],
    ) -> Tuple[Float[Tensor, "B K"], Float[Tensor, "B P K"], Float[Tensor, "B P"], Any]:
        B, P = angles.shape[0], barycentric_coords.shape[0]
        assert scales.shape[0] == B and vert_idx.shape[0] == P
        assert barycentric_coords.shape[1] == 3 and vert_idx.shape[1] == 3

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

        with torch.no_grad():
            eigen_vec, mass = self._eigen_vec, self._mass
            M, V, K = eigen_vec.shape

            eigen_vec = eigen_vec[:, vert_idx.view(-1)]  # M, P*3, K
            mass = mass[:, vert_idx.view(-1)].view(1, P, 3)  # 1, P, 3

        weights2 = weights.new_zeros((B, M)).scatter_(1, index=x_idx, src=weights)
        # B,M x M,K -> B,K
        evals = weights2 @ self._eigen_val
        # B,M x M,P*3,K -> B,P*3,K -> B, P, 3, K
        evecs = torch.einsum("ij,jkl->ikl", weights2, eigen_vec).view(B, P, 3, K)

        bary_W = barycentric_coords.unsqueeze(0)  # 1, P, 3

        # 1,P,1,3 x B,P,3,K -> B,P,K
        evec_interp = torch.matmul(bary_W.unsqueeze(2), evecs).squeeze(2)
        mass_interp = linalg.vecdot(bary_W, mass)  # 1, P

        return evals, evec_interp, mass_interp, (weights2,)

    def barycentric_albo_batchwise_next(
        self,
        albo_weights: Any,  # From barycentric_albo_eigenquantities
        barycentric_coords: Float[Tensor, "B 3"],
        vert_idx: Int[Tensor, "B 3"],
    ) -> Tuple[Float[Tensor, "B K"], Float[Tensor, "B"]]:
        B, M = barycentric_coords.shape[0], self._eigen_val.shape[0]

        weights2: Float[Tensor, "B M"]
        (weights2,) = albo_weights

        assert weights2.shape[0] == B and vert_idx.shape[0] == B
        assert weights2.shape[1] == M and vert_idx.shape[1] == 3

        with torch.no_grad():
            eigen_vec, mass = self._eigen_vec, self._mass
            M, V, K = eigen_vec.shape

            eigen_vec = eigen_vec[:, vert_idx.view(-1)].view(M, B, 3, K)  # M, B, 3, K
            eigen_vec = eigen_vec.permute(1, 0, 2, 3)  # B, M, 3, K
            mass = mass[:, vert_idx.view(-1)].view(B, 3)  # B, 3

        # B,M x B,M,3,K -> B,3,K
        evecs = linalg.vecdot(weights2.unsqueeze(-1).unsqueeze(-1), eigen_vec, dim=1)

        bary_W = barycentric_coords  # B, 3

        evec_interp = torch.einsum("bt,btk->bk", bary_W, evecs)
        mass_interp = linalg.vecdot(bary_W, mass)

        return evec_interp, mass_interp
