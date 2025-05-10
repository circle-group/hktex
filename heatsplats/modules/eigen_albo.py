from dataclasses import dataclass, field
import math

import torch
import torch.linalg as linalg

from tqdm import tqdm

import heatsplats
from heatsplats.utils import (
    get_anisotropic_lbo,
    compute_eig_laplacian,
    compute_mesh_laplacian,
    interpolate_barycentric_attr_from_trivertidx,
    compute_biharmonic_distance,
    BaseObject,
)
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
        distance_weighting: str = "none"

    cfg: Config

    def configure(self, mesh: Mesh):
        self._mesh = mesh

        _iso_eigen, _all_eigen, _smp_coords, _mass = self.precompute_all_eigen()

        M = _all_eigen.shape[0]

        self._mass = _mass
        self._smp_coords_cartesian = self._make_cartesian_query(
            _smp_coords[:, 0], _smp_coords[:, 1]
        )

        k_eig = self.cfg.k_eig
        self._iso_eigen_val = _iso_eigen[:k_eig].contiguous()
        self._iso_eigen_vec = _iso_eigen[k_eig:].view(-1, k_eig).contiguous()
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
            all_eigen, sampling_coords, mass = self._precompute_all_aniso_eigen()
        else:
            fpath_base = fpath.rsplit(".", 1)[0]
            precomputed_path = f"{fpath_base}_{self.cfg.precomputed_name}.pt"
            heatsplats.info(f"Loading precomputed albo eigen from {precomputed_path}")
            try:
                precomputed = torch.load(precomputed_path, weights_only=True)
                iso_eigen = precomputed["iso_eigen"]
                all_eigen = precomputed["all_eigen"]
                sampling_coords = precomputed["sampling_coords"]
                mass = precomputed["mass"]
            except (FileNotFoundError, KeyError):
                heatsplats.info(f"Precomputed albo eigen not found")
                iso_eigen = self._precompute_iso_eigen()
                all_eigen, sampling_coords, mass = self._precompute_all_aniso_eigen()
                torch.save(
                    {
                        "iso_eigen": iso_eigen,
                        "all_eigen": all_eigen,
                        "sampling_coords": sampling_coords,
                        "mass": mass,
                    },
                    precomputed_path,
                )
        return (
            iso_eigen.to(torch.float32).to(self.device),
            all_eigen.to(torch.float32).to(self.device),
            sampling_coords.to(torch.float32).to(self.device),
            mass.to(torch.float32).to(self.device),
        )

    def _precompute_all_aniso_eigen(
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

    def _precompute_iso_eigen(self) -> torch.Tensor:
        # Compute eigenvalues and eigenvectors obtained eigendecomposing
        # the Isotropic Laplacian
        lapl, mass = compute_mesh_laplacian(
            self._mesh.verts.cpu().numpy(), self._mesh.faces.cpu().numpy()
        )
        eval, evecs = compute_eig_laplacian(lapl, mass, self.cfg.k_eig)

        flat_evecs = torch.tensor(evecs).flatten()
        iso_eigen_flat = torch.cat([torch.tensor(eval), flat_evecs])
        return iso_eigen_flat

    def _make_cartesian_query(
        self,
        angles: Float[Tensor, "G"],
        scales: Float[Tensor, "G"],
        abs_sin: bool = True,
    ) -> Float[Tensor, "G 2"]:
        G = angles.shape[0]
        assert scales.shape[0] == G

        cos_angles, sin_angles = torch.cos(angles), torch.sin(angles)
        if abs_sin:
            sin_angles = sin_angles.abs()
        query_cartesian = torch.stack(
            [
                cos_angles * scales,
                sin_angles * scales,
            ],
            dim=1,
        )
        return query_cartesian

    def interpolate_anisotropies(
        self,
        angles: Float[Tensor, "G"],
        scales: Float[Tensor, "G"],
    ) -> Float[Tensor, "G M"]:
        """
        Computes the interpolation weights for the precomputed anisotropic eigenvalues
        and eigenvectors. The weights are computed according to the distance between the
        precomputed and the desired angles and anisotropies (scales).

        G: number of heat kernels
        M: number of precomputed anisotropic eigenproperties

        Returns:
            Float[Tensor, "G M"]: weights to combine precomputed albo eigenproperties.
        """
        query_cartesian = self._make_cartesian_query(angles, scales)
        diff = self._smp_coords_cartesian.unsqueeze(1) - query_cartesian.unsqueeze(0)
        squared_distance = (diff * diff).sum(-1, keepdim=True)
        dist, idx = squared_distance.topk(k=4, largest=False, dim=0)
        x_idx = idx.squeeze().t()  # B, 4
        dist = dist.squeeze().t()  # B, 4

        weights = 1.0 / torch.clamp(dist, min=1e-16)
        weights = weights / weights.sum(dim=1, keepdim=True)

        G, M = query_cartesian.shape[0], self._eigen_val.shape[0]
        albo_weights = weights.new_zeros((G, M)).scatter_(1, index=x_idx, src=weights)

        return albo_weights

    def albo_vertices(
        self,
        albo_weights: Float[Tensor, "G M"],
        vert_idx: Optional[Int[Tensor, "P"]] = None,
    ) -> Tuple[Float[Tensor, "G K"], Float[Tensor, "G P K"], Float[Tensor, "G P"]]:
        """
        Gather the eigenvalues and eigenvectors of the Anisotropic Laplacian at the
        vertices of the mesh. if 'vert_idx' is provided the values refer to a subset of
        vertices. The precomputed albo eigenproperties are weighted according to the
        'albo_weights' provided. These weights are computed with
        'interpolate_anisotropies' and proportional to the distance between the queried
        andclosest precomputed anisotropic eigenproperties.

        G: number of heat kernels
        M: number of precomputed anisotropic eigenproperties
        P: number of vertices
        K: number of eigenvalues/eigenvectors

        Args:
            albo_weights (Float[Tensor, "G M"]): weights to combine precomputed albo
                eigenproperties. The weights are computed with 'interpolate_anisotropies'

            vert_idx (Optional[Int[Tensor, "P"]], optional): Defaults to None.
                If provided, the eigenvalues and eigenvectors are computed only for the
                vertices with the provided indices.
                If None, the eigenvalues and eigenvectors are computed for all vertices.

        Returns:
            Tuple[Float[Tensor, "G K"], Float[Tensor, "G P K"], Float[Tensor, "G P"]]:
                The eigenvalues and eigenvectors of the Anisotropic Laplacian at the
                desired vertices as well as the mass vector.
        """
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
        """
        Interpolates the precomputed eigenvalues and eigenvectors of the Anisotropic
        Laplacian at the correct angle and anisotropy at any arbitrary location on
        the surface of the mesh. Locations are provided as barycentric coordinates wrt
        the vertices of the face containing each point.

        G: number of heat kernels
        P: number of points
        M: number of precomputed anisotropic eigenproperties
        K: number of eigenvalues/eigenvectors

        Args:
            albo_weights (Float[Tensor, "G M"]): weights to combine precomputed albo
                eigenproperties. The weights are computed with 'interpolate_anisotropies'

            barycentric_coords (Float[Tensor, "P 3"]): barycentric coordinates of the
                points to interpolate the eigenvalues and eigenvectors at.

            vert_idx (Int[Tensor, "P 3"]): indices of the vertices of the faces
                containing the points to interpolate the eigenvalues and eigenvectors at.

        Returns:
            Tuple[Float[Tensor, "G K"], Float[Tensor, "G P K"], Float[Tensor, "G P"]]:
                The eigenvalues and eigenvectors of the Anisotropic Laplacian
                for all heat kernels at the desired points as well as
                the corresponding mass vectors.
        """
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
        """
        Conceptuallly similar to 'barycentric_albo_points', but instead of interpolating
        the eigenvalues and eigenvectors for diffusing all heat kernels at arbitrary
        points on the surface of the mesh, it interpolates eigenvalues and eigenvectors
        only at kernel locations.
        Since kernels can be placed anywhere on the surface of the mesh, the main
        difference is in the required shapes of the inputs and outputs.

        It does not return evals as they have usually been computed with
        'barycentric_albo_points' and are not location specific.
        """
        G, M = albo_weights.shape[0], self._eigen_val.shape[0]
        assert barycentric_coords.shape[0] == G and vert_idx.shape[0] == G
        assert albo_weights.shape[1] == M
        assert barycentric_coords.shape[1] == 3 and vert_idx.shape[1] == 3

        with torch.no_grad():
            eigen_vec, mass = self._eigen_vec, self._mass
            M, _, K = eigen_vec.shape

            eigen_vec = eigen_vec[:, vert_idx.view(-1)].view(M, G, 3, K)  # M, G, 3, K
            mass = mass[:, vert_idx.view(-1)].view(G, 3)  # G, 3

        bary_W = barycentric_coords  # B, 3

        mass_interp = linalg.vecdot(bary_W, mass)

        evecs = torch.einsum("gm,mgck->gck", albo_weights, eigen_vec)  # G, 3, K
        evec_interp = torch.matmul(bary_W.unsqueeze(1), evecs).squeeze(1)  # G, K

        return evec_interp, mass_interp

    def barycentric_ilbo_evec_points(
        self,
        barycentric_coords: Float[Tensor, "P 3"],
        vert_idx: Int[Tensor, "P 3"],
    ) -> Float[Tensor, "P K"]:
        if self.cfg.distance_weighting != "none":
            return interpolate_barycentric_attr_from_trivertidx(
                vert_idx, barycentric_coords, self._iso_eigen_vec
            )
        else:
            return None

    @torch.no_grad()
    def ilbo_evec_vertices(
        self,
        vert_idx: Optional[Int[Tensor, "P"]] = None,
    ) -> Float[Tensor, "P K"]:
        if self.cfg.distance_weighting != "none":
            evecs = self._iso_eigen_vec
            if vert_idx is not None:
                evecs = evecs[vert_idx]
            return evecs

        else:
            return None

    def compute_biharmonic_weights(
        self,
        pts_iso_evecs: Float[Tensor, "P K"],
        kernel_bary: Float[Tensor, "G 3"],
        kernel_vert_idx: Float[Tensor, "G 3"],
    ) -> Float[Tensor, "B P+1"]:
        if self.cfg.distance_weighting != "none":
            iso_evals = self.iso_evals
            kernel_iso_evecs = self.barycentric_ilbo_evec_points(
                kernel_bary, kernel_vert_idx
            )
            pts_kernel_dist: Float[Tensor, "B P"] = compute_biharmonic_distance(
                pts_iso_evecs, kernel_iso_evecs, iso_evals, pairwise=True
            )

            if self.cfg.distance_weighting == "inverse":
                weights = 1.0 / torch.clamp(pts_kernel_dist, min=1e-16)
                weights = weights / weights.sum(dim=0, keepdim=True)

            elif "gaussian" in self.cfg.distance_weighting:
                std = float(self.cfg.distance_weighting.split("_")[-1])
                assert std > 0, "Standard deviation must be positive"
                weights = torch.exp(-(pts_kernel_dist**2) / (2 * std**2))

            else:
                raise ValueError(
                    f"Unknown distance weighting: {self.cfg.distance_weighting}"
                )

            weights: Float[Tensor, "B P+1"] = torch.cat(
                (weights, torch.ones((weights.shape[0], 1), device=weights.device)),
                dim=1,
            )
            return weights
        else:
            return None

    @property
    def iso_evals(self) -> Float[Tensor, "K"]:
        return self._iso_eigen_val
