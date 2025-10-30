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
    align_eigen,
    compute_principal_curvatures,
    compute_aligned_frame,
    interpolate_barycentric_attr_from_trivertidx,
    compute_biharmonic_distance,
    compute_biharmonic_distance_pairwise,
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
        mass_type: str = "kde"  # Need to be the same as model.mass_type
        local_frames: str = "principal_curvatures"

    cfg: Config

    def configure(self, mesh: Mesh):
        self._mesh = mesh

        _iso_eigen, _all_eigen, _smp_coords, _mass, _local_direction = (
            self.precompute_all_eigen()
        )

        M = _all_eigen.shape[0]

        self._mass = _mass
        self._local_direction = _local_direction

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
                f"Eigen Albo Interpolation requires mesh path when using precomputed, "
                f"falling back to non-precomputed"
            )

        # Essentially just a wrapper for _precompute_all_eigen which makes sure
        # that the precomputed values are saved and loaded if possible
        if fpath is None or not self.cfg.use_precomputed:
            iso_eigen, iso_evecs = self._precompute_iso_eigen()
            local_direction = self._compute_local_directions()
            all_eigen, sampling_coords, mass = self._precompute_all_aniso_eigen(
                iso_evecs, local_direction
            )
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
                local_direction = precomputed["local_direction"]
            except (FileNotFoundError, KeyError):
                heatsplats.info(f"Precomputed albo eigen not found")
                iso_eigen, iso_evecs = self._precompute_iso_eigen()
                local_direction = self._compute_local_directions()
                all_eigen, sampling_coords, mass = self._precompute_all_aniso_eigen(
                    iso_evecs, local_direction
                )
                torch.save(
                    {
                        "iso_eigen": iso_eigen,
                        "all_eigen": all_eigen,
                        "sampling_coords": sampling_coords,
                        "mass": mass,
                        "local_direction": local_direction,
                    },
                    precomputed_path,
                )
        # Add isotropic eigen as first entry of all_eigen
        all_eigen = torch.cat([iso_eigen.unsqueeze(0), all_eigen], dim=0)
        sampling_coords = torch.cat(
            [torch.tensor([[0.0, 1.0]]), sampling_coords], dim=0
        )

        return (
            iso_eigen.to(torch.float32).to(self.device),
            all_eigen.to(torch.float32).to(self.device),
            sampling_coords.to(torch.float32).to(self.device),
            mass.to(torch.float32).to(self.device),
            local_direction.to(torch.float32).to(self.device),
        )

    def _compute_local_directions(self) -> torch.Tensor:
        if "axis_aligned" in self.cfg.local_frames:
            iterations = int(self.cfg.local_frames.split("_")[-1])
            local_direction, _, _ = compute_aligned_frame(
                self._mesh.verts, self._mesh.faces, self._mesh.vnorms, iterations
            )
        elif self.cfg.local_frames == "principal_curvatures":
            local_direction, _ = compute_principal_curvatures(
                self._mesh.verts.cpu().numpy(), self._mesh.faces.cpu().numpy()
            )
        else:
            raise ValueError(f"Unknown local frames: {self.cfg.local_frames}")
        return local_direction

    def _precompute_all_aniso_eigen(
        self,
        evecs_base: Optional[torch.Tensor] = None,
        local_direction: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_eigen = []
        sampling_coords = []

        evecs_base_prev = None
        # Compute eigenvalues and eigenvectors obtained eigendecomposing
        # the Anisotropic Laplacian for different rotations and anisotropies
        heatsplats.info("Computing all eigendecompositions")
        for angle in tqdm(range(0, 181, self.cfg.precompute_angles_every_deg)):
            angle = math.radians(angle)
            for i, scale in enumerate(self.cfg.precompute_anisotropies):
                sampling_coords.append(torch.tensor([angle, scale]))

                lapl, mass = get_anisotropic_lbo(
                    self._mesh.verts,
                    self._mesh.faces.T,
                    self._mesh.fnorms,
                    rotation_angle=angle,
                    anisotropy=float(scale),
                    local_direction=local_direction,
                )

                eval, evecs = compute_eig_laplacian(lapl, mass, self.cfg.k_eig)

                if evecs_base is not None:
                    evecs, eval = align_eigen(
                        evecs_base, evecs, eval, mass, align_rotation=True
                    )
                    evecs_base = evecs
                    if i == 0:
                        evecs_base_prev = evecs

                flat_evecs = torch.tensor(evecs).flatten()
                all_eigen.append(torch.cat([torch.tensor(eval), flat_evecs]))

            evecs_base = evecs_base_prev

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
        return iso_eigen_flat, evecs

    def _make_cartesian_query(
        self,
        angles: Float[Tensor, "G"],
        scales: Float[Tensor, "G"],
        abs_sin: bool = True,
    ) -> Float[Tensor, "G 2"]:
        G = angles.shape[0]
        assert scales.shape[0] == G

        # Map scale to a log space to prevent high anisotropies from dominating
        scales = torch.log1p(scales)

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
            Tuple[Float[Tensor, "G K"], Float[Tensor, "G P K"], Float[Tensor, "1 P"]]:
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

            vert_idx_flat = vert_idx.flatten()
            eigen_vec = eigen_vec[:, vert_idx_flat].view(M, P, 3, K)  # M, P, 3, K
            mass = mass[:, vert_idx_flat].view(1, P, 3)  # 1, P, 3

        # G,M x M,K -> G,K
        evals = albo_weights @ self._eigen_val
        mass_interp = torch.sum(mass * barycentric_coords, dim=-1)  # 1, P

        bary_W = barycentric_coords.view(1, P, 1, 3)
        # 1,P,1,3 x M,P,3,K -> M,P,K
        evecs = torch.matmul(bary_W, eigen_vec).squeeze(2)

        # G,M x M,P,K -> G,P,K
        # evec_interp = torch.einsum("gm,mpk->gpk", albo_weights, evecs)
        evec_interp = (albo_weights @ evecs.reshape(M, -1)).view(G, P, K)

        return evals, evec_interp, mass_interp

    def barycentric_albo_points_old(
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
            Tuple[Float[Tensor, "G K"], Float[Tensor, "G P K"], Float[Tensor, "1 P"]]:
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

            if self.cfg.mass_type == "interpolated":
                mass = mass[:, vert_idx.view(-1)].view(1, P, 3)  # 1, P, 3

        # G,M x M,K -> G,K
        evals = albo_weights @ self._eigen_val
        # G,M x M,P*3,K -> G,P*3,K -> G, P, 3, K
        evecs = torch.einsum("ij,jkl->ikl", albo_weights, eigen_vec).view(G, P, 3, K)

        bary_W = barycentric_coords.unsqueeze(0)  # 1, P, 3

        # 1,P,1,3 x B,P,3,K -> B,P,K
        evec_interp = torch.matmul(bary_W.unsqueeze(2), evecs).squeeze(2)

        if self.cfg.mass_type == "interpolated":
            mass_interp = linalg.vecdot(bary_W, mass)  # 1, P
        else:
            mass_interp = None

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

            if self.cfg.mass_type == "interpolated":
                mass = mass[:, vert_idx.view(-1)].view(G, 3)  # G, 3

        bary_W = barycentric_coords  # B, 3

        if self.cfg.mass_type == "interpolated":
            mass_interp = linalg.vecdot(bary_W, mass)
        else:
            mass_interp = None

        evecs = torch.einsum("gm,mgck->gck", albo_weights, eigen_vec)  # G, 3, K
        evec_interp = torch.matmul(bary_W.unsqueeze(1), evecs).squeeze(1)  # G, K

        return evec_interp, mass_interp

    def barycentric_ilbo_evec_points(
        self,
        barycentric_coords: Float[Tensor, "P 3"],
        vert_idx: Int[Tensor, "P 3"],
    ) -> Float[Tensor, "P K"]:
        if self.cfg.distance_weighting != "none" or self.cfg.mass_type == "kde":
            return interpolate_barycentric_attr_from_trivertidx(
                vert_idx, barycentric_coords, self._iso_eigen_vec
            )
        else:
            return None

    def barycentric_local_directions_gaussians(
        self,
        barycentric_coords: Float[Tensor, "G 3"],
        vert_idx: Int[Tensor, "G 3"],
    ) -> Float[Tensor, "G 3"]:
        return interpolate_barycentric_attr_from_trivertidx(
            vert_idx, barycentric_coords, self._local_direction
        )

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

    def compute_pts2kernel_biharmonic_distance(
        self,
        pts_iso_evecs: Float[Tensor, "P K"],
        kernel_bary: Float[Tensor, "G 3"],
        kernel_vert_idx: Float[Tensor, "G 3"],
    ) -> Float[Tensor, "G P"]:
        iso_evals = self.iso_evals
        with torch.profiler.record_function("barycentric_ilbo_evec_points"):
            kernel_iso_evecs = self.barycentric_ilbo_evec_points(
                kernel_bary, kernel_vert_idx
            )
        with torch.profiler.record_function("compute_biharmonic_distance"):
            pts_kernel_dist: Float[Tensor, "G P"] = (
                compute_biharmonic_distance_pairwise(
                    pts_iso_evecs, kernel_iso_evecs, iso_evals, triton=True
                )
            )
        return pts_kernel_dist

    def compute_biharmonic_weights(
        self,
        pts_iso_evecs: Float[Tensor, "P K"],
        kernel_bary: Float[Tensor, "G 3"],
        kernel_vert_idx: Float[Tensor, "G 3"],
        pts2kernel_dist: Optional[Float[Tensor, "G P"]] = None,
    ) -> Float[Tensor, "G P+1"]:
        if self.cfg.distance_weighting == "none":
            return None

        if pts2kernel_dist is None:
            pts2kernel_dist = self.compute_pts2kernel_biharmonic_distance(
                pts_iso_evecs, kernel_bary, kernel_vert_idx
            )

        if self.cfg.distance_weighting == "inverse":
            weights = 1.0 / torch.clamp(pts2kernel_dist, min=1e-16)
            weights = weights / weights.sum(dim=0, keepdim=True)

        elif "gaussian" in self.cfg.distance_weighting:
            std = float(self.cfg.distance_weighting.split("_")[-1])
            assert std > 0, "Standard deviation must be positive"
            weights = torch.exp(-(pts2kernel_dist**2) / (2 * std**2))

        else:
            raise ValueError(
                f"Unknown distance weighting: {self.cfg.distance_weighting}"
            )

        ones_col = weights.new_ones((weights.shape[0], 1))
        weights: Float[Tensor, "G P+1"] = torch.cat(
            (weights, ones_col),
            dim=1,
        )
        return weights

    @torch.no_grad()
    def compute_biharmonic_dist_kde_mass(
        self,
        pts_iso_evecs: Float[Tensor, "P K"],
        kernel_bary: Float[Tensor, "G 3"],
        kernel_vert_idx: Float[Tensor, "G 3"],
        pts2kernel_dist: Optional[Float[Tensor, "G P"]] = None,
        sigma: float = None,
        total_area_normalise: bool = True,
    ) -> Float[Tensor, "G P+1"]:
        iso_evals = self.iso_evals

        if pts2kernel_dist is None:
            pts2kernel_dist = self.compute_pts2kernel_biharmonic_distance(
                pts_iso_evecs, kernel_bary, kernel_vert_idx
            )

        pts2pts_dist: Float[Tensor, "P P"] = compute_biharmonic_distance_pairwise(
            pts_iso_evecs, pts_iso_evecs, iso_evals, triton=True
        )

        G, P = pts2kernel_dist.shape

        # Initialize the distances matrix with zeros
        # The diagonal for the kernels (self-distance) is 0 (already initialized)
        distances = torch.zeros((G, P + 1, P + 1), device=self.device)

        # Fill the top-left P x P block with pts2pts_dist (same for all G)
        distances[:, :P, :P] = pts2pts_dist.unsqueeze(0)

        # Fill the last row (kernels to points) and last column (points to kernels)
        distances[:, :P, P] = pts2kernel_dist  # Gaussian to points
        distances[:, P, :P] = pts2kernel_dist  # Points to Gaussian

        N = P + 1

        if sigma is None:
            # Mask out zeros on the diagonal to compute median of non-zero distances
            mask = ~torch.eye(N, device=self.device, dtype=torch.bool).unsqueeze(0)
            masked_distances = distances[mask.expand(G, -1, -1)].view(G, N * (N - 1))
            # Median over non-diagonal distances per batch
            sigma = masked_distances.median(dim=1).values  # [G]
        else:
            # Use scalar sigma and broadcast to all batches
            sigma = torch.full((G,), sigma, device=self.device, dtype=distances.dtype)

        # Compute Gaussian kernel matrix: K[b, i, j] = exp(-D[g, i, j]^2 / sigma[g]^2)
        sigma2 = sigma.view(G, 1, 1) ** 2  # [G, 1, 1]
        K = torch.exp(-(distances**2) / sigma2)  # [G, N, N]

        # Estimate density rho[g, i] = sum_j K[g, i, j]
        rho = K.sum(dim=2)  # [G, N] => [G, P+1]

        # Inverse density as an approximation of mass
        mass: Float[Tensor, "G P+1"] = 1.0 / (rho + 1e-12)  # [G, P+1]

        if total_area_normalise:
            # Normalize per batch to sum up to total_area
            mass = mass * (self._mesh.tot_area / mass.sum(dim=1, keepdim=True))

        return mass

    @property
    def iso_evals(self) -> Float[Tensor, "K"]:
        return self._iso_eigen_val
