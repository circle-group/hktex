import torch
import igl

import numpy as np
from typing import Tuple

from heatsplats.utils.typing import *

__all__ = [
    "compute_aligned_frame",
    "compute_principal_curvatures",
]


def compute_principal_curvatures(
    np_pos: Float[np.ndarray, "V 3"], np_faces: Int[np.ndarray, "F 3"]
) -> Tuple[Float[Tensor, "V 3"], Float[Tensor, "V 3"]]:
    pd1, pd2, *_ = igl.principal_curvature(np_pos, np_faces)
    return torch.from_numpy(pd1), torch.from_numpy(pd2)


def get_adjacency(
    num_verts: int, faces: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)

    sym_edges = torch.cat([edges, edges.flip(1)], dim=0)
    edge_index = torch.unique(sym_edges, dim=0).T
    degrees = torch.bincount(edge_index[0], minlength=num_verts).float()
    degrees[degrees == 0] = 1.0

    return edge_index, degrees


def compute_aligned_frame(
    verts: torch.Tensor,
    faces: torch.Tensor,
    normals: torch.Tensor,
    num_iterations: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes a smooth, axis-aligned local reference frame for each vertex of a mesh.

    Args:
        verts (torch.Tensor): Tensor of vertex positions, shape (N, 3).
        faces (torch.Tensor): Tensor of face indices, shape (F, 3).
        normals (torch.Tensor): Tensor of vertex normals, shape (N, 3).
        num_iterations (int): The number of smoothing iterations.

    Returns:
        t1 (torch.Tensor): The primary tangent vector for each vertex, shape (N, 3).
        t2 (torch.Tensor): The secondary tangent vector for each vertex, shape (N, 3).
        n (torch.Tensor): The input normal vector for each vertex, shape (N, 3).
    """
    device = verts.device
    num_verts = verts.shape[0]

    # === Step 1: Create an Initial "Best-Guess" Aligned Field ===

    # Global axes (X, Y, Z)
    global_axes = torch.eye(3, device=device, dtype=verts.dtype)

    # Find the best global axis for each normal (the one most perpendicular)
    # This corresponds to the axis with the minimum absolute dot product.
    abs_dot_products = torch.abs(normals @ global_axes.T)
    best_axis_indices = torch.argmin(abs_dot_products, dim=1)
    best_axes = global_axes[best_axis_indices]

    # Project the best axis onto the tangent plane for each vertex
    dot_prod = torch.einsum("ni,ni->n", best_axes, normals)
    projection_on_normal = dot_prod.unsqueeze(1) * normals
    initial_tangents = best_axes - projection_on_normal
    initial_tangents = torch.nn.functional.normalize(initial_tangents, p=2, dim=1)

    # === Step 2: Represent the Field as Tensors ===

    tensors = torch.einsum("ni,nj->nij", initial_tangents, initial_tangents)

    # === Step 3: Smooth the Tensor Field ===

    # Get mesh adjacency info for smoothing
    edge_index, degrees = get_adjacency(num_verts, faces)
    edge_index = edge_index.to(device)
    degrees = degrees.to(device)

    # The smoothing is done by iterative averaging over neighbors
    for _ in range(num_iterations):
        # Gather neighbor tensors
        source_tensors = tensors[edge_index[1]]  # Tensors of all source vertices

        # Sum neighbor tensors for each target vertex
        summed_neighbor_tensors = torch.zeros_like(tensors)
        summed_neighbor_tensors.index_add_(0, edge_index[0], source_tensors)
        tensors = summed_neighbor_tensors / degrees.view(-1, 1, 1)

    # === Step 4: Extract the Final Smoothed Frame ===

    # Find the principal eigenvector of the smoothed tensor field
    # torch.linalg.eigh returns eigenvalues in ascending order, so the last
    # eigenvector corresponds to the largest eigenvalue.
    try:
        _, eigenvectors = torch.linalg.eigh(tensors)
        t_prime = eigenvectors[:, :, -1]
    except torch.linalg.LinAlgError as e:
        print(
            f"Warning: Eigenvalue decomposition failed. Returning initial tangents. Error: {e}"
        )
        # Fallback to a less robust method if SVD fails (e.g., on degenerate tensors)
        t_prime = initial_tangents

    # Project the result back onto the tangent plane to ensure orthogonality
    dot_prod_final = torch.einsum("ni,ni->n", t_prime, normals)
    projection_final = dot_prod_final.unsqueeze(1) * normals
    t1 = t_prime - projection_final
    t1 = torch.nn.functional.normalize(t1, p=2, dim=1)

    # The second tangent is the cross product of the normal and the first tangent
    t2 = torch.cross(normals, t1, dim=1)

    return t1, t2, normals
