import math
import torch
import torch_geometric.nn
import numpy as np

from typing import Optional, Tuple
from torch_geometric.utils import scatter
from tqdm import tqdm

import utils


class EigenAlboInterpolation:
    SVD_DRIVER = "gesvda"

    def __init__(self, verts, faces, fnorm, k_eig=256, fpath=None, device="cpu"):
        self._verts = verts
        self._faces = faces
        self._fnorm = fnorm
        self._device = device

        self._k_eig = k_eig
        _all_eigen, _smp_coords, _mass = self.precompute_all_eigen(fpath)

        # self._all_eigen = _all_eigen
        self._mass = _mass
        self._smp_coords_cartesian = torch.stack(
            [
                torch.cos(_smp_coords[:, 0]) * _smp_coords[:, 1],
                torch.sin(_smp_coords[:, 0]) * _smp_coords[:, 1],
            ],
            dim=1,
        )

        self._eigen_val = _all_eigen[:, :k_eig].contiguous()
        self._eigen_vec = (
            _all_eigen[:, k_eig:]
            .view(-1, self._verts.shape[0], self._k_eig)
            .contiguous()
        )

    def precompute_all_eigen(
        self, fpath: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Essentially just a wrapper for _precompute_all_eigen which makes sure
        # that the precomputed values are saved and loaded if possible
        if fpath is None:
            all_eigen, sampling_coords, mass = self._precompute_all_eigen()
        else:
            fformat = "." + fpath.split(".")[-1]
            eigen_path = fpath.replace(fformat, "_all_eigen.pt")
            smp_coords_path = fpath.replace(fformat, "_smp_coords.pt")
            mass_path = fpath.replace(fformat, "_mass.pt")
            try:
                all_eigen = torch.load(eigen_path)
                sampling_coords = torch.load(smp_coords_path)
                mass = torch.load(mass_path)
            except (FileNotFoundError, AssertionError):
                all_eigen, sampling_coords, mass = self._precompute_all_eigen()
                torch.save(all_eigen, eigen_path)
                torch.save(sampling_coords, smp_coords_path)
                torch.save(mass, mass_path)
        return (
            all_eigen.to(torch.float32).to(self._device),
            sampling_coords.to(torch.float32).to(self._device),
            mass.to(torch.float32).to(self._device),
        )

    def _precompute_all_eigen(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_eigen = []
        sampling_coords = []

        # Compute eigenvalues and eigenvectors obtained eigendecomposing
        # the Anisotropic Laplacian for different rotations and anisotropies
        print("> Precomputing all eigendecompositions")
        for angle in tqdm(range(0, 180, 30)):
            angle = math.radians(angle)
            for scale in [1, 2.5, 5, 7.5, 10, 25, 50, 75, 100]:
                sampling_coords.append(torch.tensor([angle, scale]))

                lapl, mass = utils.get_anisotropic_lbo(
                    torch.tensor(self._verts),
                    torch.tensor(self._faces).T,
                    torch.tensor(self._fnorm),
                    rotation_angle=angle,
                    anisotropy=float(scale),
                )

                eval, evecs = utils.compute_eig_laplacian(lapl, mass, self._k_eig)

                flat_evecs = torch.tensor(evecs).flatten()
                all_eigen.append(torch.cat([torch.tensor(eval), flat_evecs]))

        mass = torch.tensor(mass).unsqueeze(0).contiguous()

        polar_smp_coords = torch.stack(sampling_coords)

        return torch.stack(all_eigen), polar_smp_coords, mass

    def get_albo_eigenquantities_old(
        self, angles: torch.Tensor, scales: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Should be on current device already .to(self._device)
        query = torch.stack([angles, scales], dim=1)
        query_cartesian = torch.stack(
            [
                torch.cos(query[:, 0]) * query[:, 1],
                torch.sin(query[:, 0]) * query[:, 1],
            ],
            dim=1,
        )

        with torch.no_grad():
            diff = self._smp_coords_cartesian.unsqueeze(1) - query_cartesian.unsqueeze(
                0
            )
            squared_distance = (diff * diff).sum(-1, keepdim=True)
            idx = squared_distance.topk(k=4, largest=False, dim=0)[1]
            y_idx = torch.arange(idx.size(1), device=query.device).repeat_interleave(
                idx.size(0)
            )
            x_idx = idx.squeeze().t().reshape(-1)

            closest_cartesian = self._smp_coords_cartesian[x_idx]

        diff = query_cartesian[y_idx] - closest_cartesian
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

        y = scatter(
            self._all_eigen[x_idx] * weights,
            y_idx,
            0,
            query.size(0),
            reduce="sum",
        )
        y = y / scatter(weights, y_idx, 0, query.size(0), reduce="sum")

        albo_eigenquantities = y
        evals = albo_eigenquantities[:, : self._k_eig].contiguous()
        evecs = albo_eigenquantities[:, self._k_eig :].view(
            -1, self._verts.shape[0], self._k_eig
        )
        evecs = evecs.contiguous()
        # evecs = utils.stiefel_projx(evecs, driver=self.SVD_DRIVER)
        return evals, evecs, self._mass

    def get_albo_eigenquantities(
        self, angles: torch.Tensor, scales: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        weights = weights.unsqueeze(-1)
        evals = (
            weights
            * self._eigen_val[x_idx.flatten()].view(
                *x_idx.shape, *self._eigen_val.shape[1:]
            )
        ).sum(dim=1, keepdim=False)
        evecs = (
            weights.unsqueeze(-1)
            * self._eigen_vec[x_idx.flatten()].view(
                *x_idx.shape, *self._eigen_vec.shape[1:]
            )
        ).sum(dim=1, keepdim=False)

        # x_idx = x_idx.flatten()
        # weights = weights.reshape(-1, 1)
        # evals = scatter(self._eigen_val[x_idx] * weights, y_idx, 0, B, reduce="sum")
        # evecs = scatter(self._eigen_vec[x_idx] * weights.unsqueeze(-1), y_idx, 0, B, reduce="sum")

        return evals, evecs, self._mass


if __name__ == "__main__":
    import trimesh
    import numpy as np

    mesh_path = "../objects/spot_triangulated.obj"
    mesh = utils.load_mesh(mesh_path, show=False)

    v, f = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    # v, f = trimesh.remesh.subdivide(v, f)
    mesh = trimesh.Trimesh(v, f)

    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    fnorm = np.array(mesh.face_normals)

    pca_eigen_albo = EigenAlboInterpolation(
        verts, faces, fnorm, k_eig=256, fpath=mesh_path, device="cuda"
    )

    # albo_evals, albo_evecs, mass = pca_eigen_albo.get_albo_eigenquantities(
    #     angle=45.0, scale=33.0
    # )

    albo_evals, albo_evecs, mass = pca_eigen_albo.get_albo_eigenquantities(
        angles=torch.deg2rad(torch.tensor([45.0, 18.3, 10.0])),
        scales=torch.tensor([33.0, 60, 5.2]),
    )

    colours = torch.zeros([3, *verts.shape])
    # idxs = torch.randint(0, mesh.vertices.shape[0], (3,))
    idxs = torch.tensor([3804, 0, 4274])
    # colours[:, idxs, :] = torch.tensor(
    #     [1.0, 0, 0], dtype=torch.float64
    # ).unsqueeze(0)
    # colours[:, 0, :] = torch.tensor([1.0, 0, 0]).unsqueeze(0)
    colours[torch.arange(3), idxs, :] = torch.tensor(
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=torch.float
    )

    colours = utils.heat_diffusion(
        colours, mass, albo_evals, albo_evecs, torch.tensor([0.001, 0.1, 0.01])
    )

    colours = colours.sum(dim=0)

    colours = (colours - colours.min()) / (colours.max() - colours.min())
    colours *= 255
    colours = colours.squeeze().numpy()

    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colours)
