import math
import torch
import torch_geometric.nn
import geoopt

from typing import Optional, Tuple
from torch_geometric.utils import scatter

import utils


class EigenAlboInterpolation:
    def __init__(self, verts, faces, fnorm, k_eig=256, fpath=None):
        self._verts = verts
        self._faces = faces
        self._fnorm = fnorm

        self._k_eig = k_eig
        self._all_eigen, self._smp_coords, self._mass = (
            self.precompute_all_eigen(fpath)
        )
        self._stiefel_manifold = geoopt.Stiefel()

    def precompute_all_eigen(
        self, fpath: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Essentially just a wrapper for _precompute_all_eigen which makes sure
        # that the precomputed values are saved and loaded if possible

        if fpath is None:
            return self._precompute_all_eigen(fpath)

        else:
            eigen_path = fpath.replace(".obj", "_all_eigen.pt")
            smp_coords_path = fpath.replace(".obj", "_smp_coords.pt")
            mass_path = fpath.replace(".obj", "_mass.pt")
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
            all_eigen.to(torch.float32),
            sampling_coords.to(torch.float32),
            mass.to(torch.float32),
        )

    def _precompute_all_eigen(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_eigen = []
        sampling_coords = []

        # Compute eigenvalues and eigenvectors obtained eigendecomposing
        # the Anisotropic Laplacian for different rotations and anisotropies
        for angle in range(0, 180, 30):
            for scale in [1, 2.5, 5, 7.5, 10, 25, 50, 75, 100]:
                sampling_coords.append(torch.tensor([angle, scale]))

                lapl, mass = utils.get_anisotropic_lbo(
                    torch.tensor(self._verts),
                    torch.tensor(self._faces).T,
                    torch.tensor(self._fnorm),
                    rotation_angle=math.radians(angle),
                    anisotropy=float(scale),
                )

                eval, evecs = utils.compute_eig_laplacian(
                    lapl, mass, self._k_eig
                )

                flat_evecs = torch.tensor(evecs).flatten()
                all_eigen.append(torch.cat([torch.tensor(eval), flat_evecs]))

        mass = torch.tensor(mass).unsqueeze(0).contiguous()

        polar_smp_coords = torch.stack(sampling_coords)

        return torch.stack(all_eigen), polar_smp_coords, mass

    # def get_albo_eigenquantities(
    #     self, angle, scale
    # ) -> Tuple[torch.Tensor, torch.Tensor]:

    #     query = torch.tensor([angle, scale]).unsqueeze(0)

    #     # albo_eigenquantities = torch_geometric.nn.knn_interpolate(
    #     #     self._all_eigen, self._smp_coords, query, k=4
    #     # )
    #     with torch.no_grad():
    #         assign_index = torch_geometric.nn.knn(self._smp_coords, query, k=4)
    #         y_idx, x_idx = assign_index[0], assign_index[1]

    #         closest_polar = self._smp_coords[x_idx]

    #         closest_cartesian = torch.stack(
    #             [
    #                 torch.cos(closest_polar[:, 0]) * closest_polar[:, 1],
    #                 torch.sin(closest_polar[:, 0]) * closest_polar[:, 1],
    #             ],
    #             dim=1,
    #         )
    #         query_cartesian = torch.stack(
    #             [
    #                 torch.cos(query[:, 0]) * query[:, 1],
    #                 torch.sin(query[:, 0]) * query[:, 1],
    #             ],
    #             dim=1,
    #         )
    #         diff = query_cartesian - closest_cartesian
    #         squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
    #         weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

    #     y = scatter(
    #         self._all_eigen[x_idx] * weights,
    #         y_idx,
    #         0,
    #         query.size(0),
    #         reduce="sum",
    #     )
    #     y = y / scatter(weights, y_idx, 0, query.size(0), reduce="sum")

    #     albo_eigenquantities = y.squeeze(0)
    #     evals = albo_eigenquantities[: self._k_eig].unsqueeze(0).contiguous()
    #     evecs = albo_eigenquantities[self._k_eig :].reshape(
    #         self._verts.shape[0], self._k_eig
    #     )
    #     evecs = evecs.unsqueeze(0).contiguous()
    #     # evecs = self.gram_schmidt(evecs).unsqueeze(0).contiguous()
    #     evecs = self._stiefel_manifold.projx(evecs)
    #     return evals, evecs, self._mass

    def get_albo_eigenquantities(
        self, angles: torch.Tensor, scales: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        query = torch.stack([angles, scales], dim=1)
        query_cartesian = torch.stack(
            [
                torch.cos(query[:, 0]) * query[:, 1],
                torch.sin(query[:, 0]) * query[:, 1],
            ],
            dim=1,
        )

        # albo_eigenquantities = torch_geometric.nn.knn_interpolate(
        #     self._all_eigen, self._smp_coords, query, k=4
        # )
        # with torch.no_grad():
        # assign_index = torch_geometric.nn.knn(self._smp_coords, query, k=4)
        # y_idx, x_idx = assign_index[0], assign_index[1]

        diff = self._smp_coords.unsqueeze(1) - query.unsqueeze(0)
        squared_distance = (diff * diff).sum(-1, keepdim=True)
        idx = squared_distance.topk(k=4, largest=False, dim=0)[1]
        y_idx = torch.arange(idx.size(1)).repeat_interleave(idx.size(0))
        x_idx = idx.squeeze().t().reshape(-1)

        closest_polar = self._smp_coords[x_idx]

        closest_cartesian = torch.stack(
            [
                torch.cos(closest_polar[:, 0]) * closest_polar[:, 1],
                torch.sin(closest_polar[:, 0]) * closest_polar[:, 1],
            ],
            dim=1,
        )

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
        evecs = self._stiefel_manifold.projx(evecs)
        return evals, evecs, self._mass


def differentiable_knn_interpolate(x, pos_x, pos_y, k=3):
    diff = pos_x.unsqueeze(1) - pos_y.unsqueeze(0)
    squared_distance = (diff * diff).sum(-1, keepdim=True)
    idx = squared_distance.topk(k, largest=False, dim=0)[1]
    y_idx = torch.arange(idx.size(1)).repeat_interleave(idx.size(0))
    x_idx = idx.squeeze().t().reshape(-1)

    selected_squared_dist = squared_distance[x_idx, y_idx]

    weights = 1.0 / torch.clamp(selected_squared_dist, min=1e-16)

    y2 = scatter(x[x_idx] * weights, y_idx, 0, pos_y.size(0), reduce="sum")
    y2 = y2 / scatter(weights, y_idx, 0, pos_y.size(0), reduce="sum")
    return y2


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
        verts, faces, fnorm, k_eig=256, fpath=mesh_path
    )

    # albo_evals, albo_evecs, mass = pca_eigen_albo.get_albo_eigenquantities(
    #     angle=45.0, scale=33.0
    # )

    albo_evals, albo_evecs, mass = pca_eigen_albo.get_albo_eigenquantities(
        angles=torch.tensor([45.0, 18.3, 10.0]),
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
