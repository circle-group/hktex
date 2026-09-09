import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.optimize

import torch
import robust_laplacian

from .typing import *
from .local_reference_frames import compute_principal_curvatures

__all__ = [
    "get_anisotropic_lbo",
    "compute_mesh_laplacian",
    "compute_point_cloud_laplacian",
    "compute_eig_laplacian",
    "align_eigen",
]


def get_anisotropic_lbo(
    pos: torch.Tensor,
    face: torch.Tensor,
    face_normals: Optional[torch.Tensor] = None,
    rotation_angle: Optional[float] = 0.0,
    anisotropy: Optional[float] = 0.0,
    local_direction: Optional[torch.Tensor] = None,
) -> Tuple[scipy.sparse.csc_matrix, np.ndarray]:
    """
    Computes the anisotropic Laplace-Beltrami operator.

    Args:
        pos: Vertex positions, shape (N, 3).
        face: Face indices, shape (3, F).
        face_normals: Optional pre-computed face normals, accepts both (3, F) and (F, 3) shapes.
        rotation_angle: Optional rotation angle in radians.
        anisotropy: Optional anisotropy parameter.
        local_direction: Optional local direction for each vertex serving as a reference
            frame for the anisotropic diffusion tensor.

    Returns:
        A tuple containing the stiffness matrix (W) and mass matrix diagonal (A_diag).
    """
    # --- 1. Setup and Input Conversion ---
    device = pos.device
    n_verts = pos.shape[0]
    n_faces = face.shape[1]

    face = face.long()
    face_t = face.T

    face_vertices = pos[face_t]
    v0, v1, v2 = face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2]

    N = face_normals.to(device)
    N = N / torch.norm(N, dim=1, keepdim=True).clamp(min=1e-9)

    # --- 3. Anisotropic Diffusion Tensor ---
    np_pos = pos.cpu().numpy()
    np_faces_t = face_t.cpu().numpy()

    if local_direction is not None:
        Umax_vert = local_direction.to(device, dtype=torch.float32)
    else:
        pd1, _ = compute_principal_curvatures(np_pos, np_faces_t)
        Umax_vert = pd1.to(device, dtype=torch.float32)

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


def align_eigen(
    evecs_ref, evecs_to_align, evals_to_align, mass_matrix, align_rotation=True
):
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
    - mass_matrix (np.ndarray or scipy.sparse.spmatrix): The mass matrix which defines
        the inner product.

    Returns:
    - evecs_aligned (np.ndarray): The aligned eigenvector matrix.
    - evals_aligned (np.ndarray): The aligned eigenvalues.
    """
    # Find optimal permutation:
    # maximize the absolute value of the M-weighted inner product: |u_i^T @ M @ v_j|.
    # NB. The inner product u.T @ M @ v measures how much the vector v projects onto the
    # vector u in the mass weighted space. Its value is maximized when the vectors
    # are perfectly aligned.
    m_inner_products = evecs_ref.T * mass_matrix @ evecs_to_align
    cost_matrix = -np.abs(m_inner_products)

    # linear_sum_assignment finds the permutation that minimizes the total cost.
    # It returns the optimal row and column indices.
    _, permuted_indices = scipy.optimize.linear_sum_assignment(cost_matrix)

    # Reorder the eigenvectors to be aligned according to the optimal permutation.
    evecs_permuted = evecs_to_align[:, permuted_indices]

    if align_rotation:
        # Find optimal rotation using M-weighted Orthogonal Procrustes:
        # finds the rotation matrix R that minimizes ||evecs_ref - evecs_permuted @ R||_M^2,
        # where ||.||_M is the M-weighted Frobenius norm.
        # The solution is found via the SVD of the M-weighted correlation matrix.

        correlation_matrix = evecs_permuted.T * mass_matrix @ evecs_ref
        U, _, Vt = scipy.linalg.svd(correlation_matrix)
        rotation_matrix = U @ Vt

        # Apply the optimal rotation.
        evecs_rotated = evecs_permuted @ rotation_matrix
    else:
        evecs_rotated = evecs_permuted

    # Correct the signs using M-weighted inner product:
    # check the sign of the M-weighted dot product between corresponding vectors.

    # Efficiently calculate the diagonal of evecs_ref.T @ mass_matrix @ evecs_rotated
    m_dot_products = np.sum(evecs_ref * (mass_matrix[:, None] * evecs_rotated), axis=0)
    signs = np.sign(m_dot_products)

    # Multiply by the signs to ensure they point in the same direction.
    # A sign of 0 (if vectors are perfectly orthogonal) is treated as +1.
    signs[signs == 0] = 1
    evecs_aligned = evecs_rotated * signs

    # Ensure the eigenvalues are aligned with the permuted indices.
    # evals_aligned = evals_to_align[np.argsort(permuted_indices)]
    evals_aligned = evals_to_align[permuted_indices]

    return evecs_aligned, evals_aligned
