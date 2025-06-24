import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.optimize

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
    "get_anisotropic_lbo_old",
    "align_eigen",
]


def get_anisotropic_lbo_old(
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
        f_reference = torch.stack([fpd1, fpd2, face_normals.cpu()], dim=2).to(
            torch.float64
        )
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


def get_anisotropic_lbo(
    pos: torch.Tensor,
    face: torch.Tensor,
    face_normals: Optional[torch.Tensor] = None,
    rotation_angle: Optional[float] = 0.0,
    anisotropy: Optional[float] = 0.0,
    local_direction=None,
) -> Tuple[scipy.sparse.csc_matrix, np.ndarray]:
    """
    Computes the anisotropic Laplace-Beltrami operator.

    Args:
        pos: Vertex positions, shape (N, 3).
        face: Face indices, shape (3, F).
        face_normals: Optional pre-computed face normals, accepts both (3, F) and (F, 3) shapes.
        rotation_angle: Optional rotation angle in radians.
        anisotropy: Optional anisotropy parameter.

    Returns:
        A tuple containing the stiffness matrix (W) and mass matrix diagonal (A_diag).
    """
    # --- 1. Setup and Input Conversion ---
    device = pos.device
    n_verts = pos.shape[0]
    n_faces = face.shape[1]

    face_t = face.T

    face_vertices = pos[face_t]
    v0, v1, v2 = face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2]

    N = face_normals.to(device)
    N = N / torch.norm(N, dim=1, keepdim=True).clamp(min=1e-9)

    # --- 3. Anisotropic Diffusion Tensor ---
    np_pos = pos.cpu().numpy()
    np_faces_t = face_t.cpu().numpy()

    if local_direction is not None:
        pd1 = local_direction
    else:
        pd1, _, _, _ = igl.principal_curvature(np_pos, np_faces_t)
    Umax_vert = torch.from_numpy(pd1).to(device, dtype=torch.float32)

    # Interpolate vertex-based directions to faces -> shape is (F, 3)
    Umax_face = Umax_vert[face_t].mean(dim=1)

    # Project Umax onto the face's tangent plane
    Umax_face = Umax_face - torch.sum(Umax_face * N, dim=1, keepdim=True) * N
    Umax_face = Umax_face / torch.norm(Umax_face, dim=1, keepdim=True).clamp(min=1e-9)
    Umin_face_derived = torch.cross(N, Umax_face, dim=1)
    Umin_face_derived = Umin_face_derived / torch.norm(
        Umin_face_derived, dim=1, keepdim=True
    ).clamp(min=1e-9)

    # Apply Rodrigues' rotation if needed
    if rotation_angle != 0.0:
        ca = torch.cos(torch.tensor(rotation_angle, device=device))
        sa = torch.sin(torch.tensor(rotation_angle, device=device))

        def rotate_vector(vec, axis, c, s):
            # Full Rodrigues' rotation formula
            return (
                vec * c
                + torch.cross(axis, vec, dim=1) * s
                + axis * torch.sum(axis * vec, dim=1, keepdim=True) * (1 - c)
            )

        Umin_final = rotate_vector(Umin_face_derived, N, ca, sa)
        Umax_final = rotate_vector(Umax_face, N, ca, sa)
    else:
        Umin_final, Umax_final = Umin_face_derived, Umax_face

    # Define per-face diffusion tensor D
    D = torch.zeros((n_faces, 2), device=device, dtype=torch.float32)
    D[:, 0] = 1.0 / (1.0 + torch.tensor(anisotropy))
    D[:, 1] = 1.0

    # --- 4. Construct Stiffness Matrix W ---
    i_s, j_s, val_s = [], [], []

    # Loop over the three edges of each triangle
    for k in range(3):
        # Get the indices of the three vertices for this edge configuration
        p_k0 = face_t[:, k]
        p_k1 = face_t[:, (k + 1) % 3]
        p_k2 = face_t[:, (k + 2) % 3]

        # Vectors corresponding to the edges opposite vertices p_k0 and p_k1
        e1 = pos[p_k1] - pos[p_k2]
        e2 = pos[p_k0] - pos[p_k2]
        e1_norm = e1 / torch.norm(e1, dim=1, keepdim=True).clamp(min=1e-9)
        e2_norm = e2 / torch.norm(e2, dim=1, keepdim=True).clamp(min=1e-9)

        # Anisotropic dot product
        term1 = torch.sum(e1_norm * Umin_final, dim=1) * torch.sum(
            e2_norm * Umin_final, dim=1
        )
        term2 = torch.sum(e1_norm * Umax_final, dim=1) * torch.sum(
            e2_norm * Umax_final, dim=1
        )
        anisotropic_dot_prod = D[:, 0] * term1 + D[:, 1] * term2

        # The sine of the angle between e1 and e2
        angle_sin = torch.norm(torch.cross(e1_norm, e2_norm, dim=1), dim=1).clamp(
            min=1e-9
        )

        weight = 0.5 * anisotropic_dot_prod / angle_sin

        # Add symmetric off-diagonal entries
        i_s.extend([p_k0, p_k1])
        j_s.extend([p_k1, p_k0])
        val_s.extend([-weight, -weight])

    W = scipy.sparse.csc_matrix(
        (
            torch.cat(val_s).cpu().numpy(),
            (torch.cat(i_s).cpu().numpy(), torch.cat(j_s).cpu().numpy()),
        ),
        shape=(n_verts, n_verts),
    )
    W.setdiag(W.diagonal() - W.sum(axis=1).A1)

    # --- 5. Construct Mass Matrix Diagonal ---
    tri_areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
    area_indices = torch.flatten(face_t)  # Flattening (F, 3) tensor
    area_vals = tri_areas.repeat_interleave(3) / 3.0

    vertex_areas = torch.zeros(n_verts, device=device).scatter_add_(
        0, area_indices, area_vals
    )
    A_diag = vertex_areas.cpu().numpy()

    return W, A_diag


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
                lapl_eigsh.astype(np.float32),
                k=k_eig,
                M=mass_mat.astype(np.float32),
                sigma=eigs_sigma,
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


def align_eigen(evecs_ref, evecs_to_align, evals_to_align, align_rotation=True):
    """
    Aligns a set of eigenvectors and eigenvalues to a reference set.

    This function solves the sign and permutation ambiguities between two sets
    of eigenvectors. It first finds the optimal ordering using the Hungarian
    algorithm (linear_sum_assignment) and then finds the optimal rotation
    using Orthogonal Procrustes analysis (via SVD).

    Parameters:
    - evecs_ref (np.ndarray): The reference eigenvector matrix (shape n_verts x k).
    - evecs_to_align (np.ndarray): The eigenvector matrix to align (shape n_verts x k).
    - evals_to_align (np.ndarray): The eigenvalues corresponding to evecs_to_align (shape k).

    Returns:
    - evecs_aligned (np.ndarray): The aligned eigenvector matrix.
    - evals_aligned (np.ndarray): The aligned eigenvalues.
    """
    # --- Step 1: Find the optimal permutation using the Hungarian algorithm ---
    # The cost matrix measures the squared Euclidean distance between all pairs of
    # eigenvectors from the two sets.
    cost_matrix = np.sum(
        (evecs_ref[:, :, np.newaxis] - evecs_to_align[:, np.newaxis, :]) ** 2, axis=0
    )

    # linear_sum_assignment finds the permutation that minimizes the total cost.
    # It returns the optimal row and column indices.
    _, permuted_indices = scipy.optimize.linear_sum_assignment(cost_matrix)

    # Reorder the eigenvectors to be aligned according to the optimal permutation.
    evecs_permuted = evecs_to_align[:, permuted_indices]

    if align_rotation:
        # --- Step 2: Find the optimal rotation using Orthogonal Procrustes ---
        # This method finds the rotation matrix R that minimizes
        # ||evecs_ref - evecs_permuted @ R||^2.
        correlation_matrix = evecs_permuted.T @ evecs_ref
        U, _, Vt = scipy.linalg.svd(correlation_matrix)
        rotation_matrix = U @ Vt

        # Apply the optimal rotation.
        evecs_rotated = evecs_permuted @ rotation_matrix
    else:
        evecs_rotated = evecs_permuted

    # --- Step 3: Correct the signs ---
    # After global rotation, individual eigenvectors might still be flipped.
    # We check the sign of the dot product between corresponding vectors.
    signs = np.sign(np.sum(evecs_ref * evecs_rotated, axis=0))

    # Multiply by the signs to ensure they point in the same direction.
    # A sign of 0 (if vectors are perfectly orthogonal) is treated as +1.
    signs[signs == 0] = 1
    evecs_aligned = evecs_rotated * signs

    # Ensure the eigenvalues are aligned with the permuted indices.
    evals_aligned = evals_to_align[[np.argsort(permuted_indices)]]

    return evecs_aligned, evals_aligned[0, :]
