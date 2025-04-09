import numpy as np
import scipy.sparse

import torch

import igl
import robust_laplacian
from torch_geometric.utils import add_self_loops, scatter, to_undirected

from .typing import *
from .misc import sparse_torch_to_np

__all__ = [
    "get_anisotropic_lbo",
    "compute_mesh_laplacian",
    "compute_point_cloud_laplacian",
    "compute_eig_laplacian",
]


def get_anisotropic_lbo(
    pos: torch.Tensor,
    face: torch.Tensor,
    face_normals: Optional[torch.Tensor] = None,
    rotation_angle: Optional[float] = 0.0,
    anisotropy: Optional[float] = 0.0,
) -> Tuple[scipy.sparse.csc_matrix, np.ndarray]:
    assert pos.size(1) == 3 and face.size(0) == 3

    num_nodes = pos.shape[0]

    def get_lapl_weights(
        left: torch.Tensor,
        centre: torch.Tensor,
        right: torch.Tensor,
        an_mat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        left_pos, central_pos, right_pos = pos[left], pos[centre], pos[right]
        left_vec = left_pos - central_pos
        right_vec = right_pos - central_pos
        if an_mat is None:
            dot = torch.einsum("ij, ij -> i", left_vec, right_vec)
        else:
            dot = torch.matmul(
                right_vec.unsqueeze(1),
                torch.matmul(an_mat, left_vec.unsqueeze(-1)),
            ).squeeze()
        cross = torch.norm(torch.cross(left_vec, right_vec, dim=1), dim=1)
        cot = dot / cross  # cot = cos / sin
        return cot / 2.0  # by definition

    if anisotropy != 0 or rotation_angle != 0:
        assert anisotropy > 0

        np_faces_t = face.cpu().numpy().T
        pd1, pd2, _, _ = igl.principal_curvature(pos.cpu().numpy(), np_faces_t)
        fpd1 = torch.tensor(igl.average_onto_faces(np_faces_t, pd1))
        fpd2 = torch.tensor(igl.average_onto_faces(np_faces_t, pd2))
        f_reference = torch.stack([fpd1, fpd2, face_normals.cpu()], dim=2)
        f_reference_t = torch.transpose(f_reference, 1, 2)
        an_scale_mat = torch.diag(torch.tensor([1 / (1 + anisotropy), 1.0, 1.0])).to(
            torch.float64
        )
        scales_mat = torch.matmul(
            torch.matmul(f_reference, an_scale_mat), f_reference_t
        )
        angle = torch.tensor(rotation_angle)
        rotation_around_normal_mat = torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle), 0],
                [torch.sin(angle), torch.cos(angle), 0],
                [0, 0, 1],
            ]
        ).to(torch.float64)
        anisotropy_mat = (
            torch.matmul(
                torch.matmul(rotation_around_normal_mat, scales_mat),
                rotation_around_normal_mat.t(),
            )
            .to(pos.device)
            .to(torch.float32)
        )
    else:
        anisotropy_mat = None

    # For each triangle face, get all three cotangents:
    w_021 = get_lapl_weights(face[0], face[2], face[1], anisotropy_mat)
    w_102 = get_lapl_weights(face[1], face[0], face[2], anisotropy_mat)
    w_012 = get_lapl_weights(face[0], face[1], face[2], anisotropy_mat)
    lapl_weight = torch.cat([w_021, w_102, w_012])

    # Face to edge:
    lapl_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    lapl_index, lapl_weight = to_undirected(lapl_index, lapl_weight)

    # Compute the diagonal part:
    deg = scatter(lapl_weight, lapl_index[0], 0, num_nodes, reduce="sum")
    edge_index, _ = add_self_loops(lapl_index, num_nodes=num_nodes)
    edge_weight = torch.cat([lapl_weight, -deg], dim=0)

    def get_areas(
        left: torch.Tensor, centre: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        central_pos = pos[centre]
        left_vec = pos[left] - central_pos
        right_vec = pos[right] - central_pos
        cross = torch.norm(torch.cross(left_vec, right_vec, dim=1), dim=1)
        area = cross / 6.0  # one-third of a triangle's area is cross / 6.0
        return area / 2.0  # since each corresponding area is counted twice

    # Like before, but here we only need the diagonal (the mass matrix):
    area_021 = get_areas(face[0], face[2], face[1])
    area_102 = get_areas(face[1], face[0], face[2])
    area_012 = get_areas(face[0], face[1], face[2])
    area_weight = torch.cat([area_021, area_102, area_012])
    area_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    area_index, area_weight = to_undirected(area_index, area_weight)
    area_deg = scatter(area_weight, area_index[0], 0, num_nodes, "sum")

    return (
        -sparse_torch_to_np(torch.sparse_coo_tensor(edge_index, edge_weight)),
        area_deg.cpu().numpy(),
    )


def compute_mesh_laplacian(
    verts: np.ndarray, faces: np.ndarray
) -> Tuple[scipy.sparse.csc_matrix, np.ndarray]:
    lapl, mass = robust_laplacian.mesh_laplacian(verts, faces)
    return lapl, mass.diagonal()


def compute_point_cloud_laplacian(
    points: np.ndarray,
) -> Tuple[scipy.sparse.csc_matrix, np.ndarray]:
    lapl, mass = robust_laplacian.point_cloud_laplacian(points)
    return lapl, mass.diagonal()


def compute_eig_laplacian(
    lapl: scipy.sparse.csc_matrix,
    massvec: np.ndarray,
    k_eig: int = 128,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the eigendecomposition of the Laplacian

    Args:
        lapl: [N x N] Laplacian
        massvec: [N] mass vector
        k_eig (int, optional): number of eigenvalues and eigenvectors desired.
            Defaults to 10.
        eps (float, optional): constant used to perturb Laplacian during
            eigendecomposition. Defaults to 1e-8.

    Raises:
        ValueError: although multiple attempts were made, the eigendecomposition
            failed.

    Returns:
        Tuple[np.ndarray, np.ndarray]: k eigenvalues, [k x N] eigenvectors.
    """

    # Prepare matrices for eigendecomposition like in DiffusionNet code
    lapl_eigsh = (lapl + scipy.sparse.identity(lapl.shape[0]) * eps).tocsc()
    mass_mat = scipy.sparse.diags(massvec)
    eigs_sigma = eps

    failcount = 0
    while True:
        try:
            evals, evecs = scipy.sparse.linalg.eigsh(
                lapl_eigsh.astype(np.float32), k=k_eig, M=mass_mat, sigma=eigs_sigma
            )
            evals = np.clip(evals, a_min=0.0, a_max=float("inf"))
            break
        except RuntimeError as exc:
            if failcount > 3:
                raise ValueError("failed to compute eigendecomp") from exc
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount))
            lapl_eigsh = lapl_eigsh + scipy.sparse.identity(lapl.shape[0]) * (
                eps * 10**failcount
            )
    return evals, evecs
