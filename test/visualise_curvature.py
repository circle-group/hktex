import sys
from pathlib import Path
import os

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))


import torch


def get_adjacency(num_verts: int, faces: torch.Tensor):
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
):
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
    dot_prod = torch.einsum("ni,ni->n", best_axes, normals)  # Batched dot product
    projection_on_normal = dot_prod.unsqueeze(1) * normals
    initial_tangents = best_axes - projection_on_normal
    initial_tangents = torch.nn.functional.normalize(initial_tangents, p=2, dim=1)

    # === Step 2: Represent the Field as Tensors ===

    # Convert each vector `d` into a tensor `T = d * d^T`
    # shape: (N, 3) -> (N, 3, 3)
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
        # index_add_ is an efficient way to perform this scatter-add operation
        summed_neighbor_tensors.index_add_(0, edge_index[0], source_tensors)

        # Average by dividing by the vertex degree
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


def compute_gframes(
    verts: torch.Tensor,
    faces: torch.Tensor,
    normals: torch.Tensor,
    scalar_field: torch.Tensor,
):
    """
    Computes a Local Reference Frame from a single scalar field, faithfully
    implementing Equations (3-6) from the GFrames paper.

    This computes the frame Lf(p) for a single function f.

    Args:
        verts (torch.Tensor): Vertex positions, shape (N, 3).
        faces (torch.Tensor): Face indices, shape (F, 3).
        normals (torch.Tensor): Vertex normals, shape (N, 3).
        scalar_field (torch.Tensor): A scalar value for each vertex, shape (N,).

    Returns:
        x_hat (torch.Tensor): The primary tangent vector, shape (N, 3).
        y_hat (torch.Tensor): The secondary tangent vector, shape (N, 3).
        z_hat (torch.Tensor): The normal vector, identical to input normals, shape (N, 3).
    """
    N = verts.shape[0]
    device = verts.device
    dtype = verts.dtype

    # 1. Calculate per-triangle gradients and areas
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    s0, s1, s2 = (
        scalar_field[faces[:, 0]],
        scalar_field[faces[:, 1]],
        scalar_field[faces[:, 2]],
    )

    e01 = v1 - v0
    e20 = v0 - v2

    face_normals = torch.cross(e01, e20, dim=1)
    face_areas = 0.5 * torch.linalg.norm(face_normals, dim=1)

    # Check for zero-area faces to avoid division by zero
    valid_faces = face_areas > 1e-8

    # Calculate gradient on each face (constant vector)
    grad_on_faces = torch.zeros_like(face_normals)

    grad_on_faces[valid_faces] = (
        s0[valid_faces].unsqueeze(1)
        * torch.cross(face_normals[valid_faces], v2[valid_faces] - v1[valid_faces])
        + s1[valid_faces].unsqueeze(1)
        * torch.cross(face_normals[valid_faces], v0[valid_faces] - v2[valid_faces])
        + s2[valid_faces].unsqueeze(1)
        * torch.cross(face_normals[valid_faces], v1[valid_faces] - v0[valid_faces])
    )
    # The formula has a 1/(2A) term, but the cross products with face_normals scale with A^2.
    # So the whole expression scales with A^2. We need to divide by (2A)^2 = (face_area * 2)^2.
    grad_on_faces[valid_faces] /= (2 * face_areas[valid_faces].unsqueeze(1)) ** 2

    # 2. Compute the area-weighted average gradient for each vertex (Equation 3)
    # Numerator: Sum of Area(t) * Grad(t) for adjacent triangles
    # Denominator: Sum of Area(t) for adjacent triangles

    weighted_grads = face_areas.unsqueeze(1) * grad_on_faces

    numerator = torch.zeros(N, 3, device=device, dtype=dtype)
    denominator = torch.zeros(N, 1, device=device, dtype=dtype)

    for i in range(3):
        numerator.index_add_(0, faces[:, i], weighted_grads)
        denominator.index_add_(0, faces[:, i], face_areas.unsqueeze(1))

    # Avoid division by zero for isolated vertices
    denominator[denominator == 0] = 1.0

    # This is x(p) from Equation (3)
    avg_gradient = numerator / denominator

    # 3. Project the averaged gradient onto the tangent plane (Equation 4)
    dot_prod = torch.einsum("ni,ni->n", avg_gradient, normals)
    projection_on_normal = dot_prod.unsqueeze(1) * normals

    # This is ˆx(p)
    x_hat = avg_gradient - projection_on_normal
    x_hat = torch.nn.functional.normalize(x_hat, p=2, dim=1)

    # 4. Construct the rest of the frame (Equations 5 & 6)
    # This is ˆz(p)
    z_hat = normals
    # This is ˆy(p)
    y_hat = torch.cross(z_hat, x_hat, dim=1)

    return x_hat, y_hat, z_hat


if __name__ == "__main__":
    import igl
    import trimesh
    import numpy as np

    import meshplot as mp
    from heatsplats import utils

    fname = "../objects/spot/spot_triangulated.obj"
    mesh = utils.load_mesh(fname)
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    v, f = mesh.vertices, mesh.faces
    avg = igl.avg_edge_length(v, f) / 2.0

    # Local reference frames based on Principla curvature ##############################
    v1, v2, k1, k2 = igl.principal_curvature(v, f)
    h2 = 0.5 * (k1 + k2)
    myplot = mp.plot(v, f, shading={"wireframe": False}, return_plot=True)

    myplot.add_lines(v + v1 * avg, v - v1 * avg, shading={"line_color": "red"})
    myplot.add_lines(v + v2 * avg, v - v2 * avg, shading={"line_color": "green"})

    # Local reference frames aligned with axes and smoothed ############################
    v3, v4, _ = compute_aligned_frame(
        verts=torch.tensor(v),
        faces=torch.tensor(f),
        normals=torch.tensor(mesh.vertex_normals),
        num_iterations=5,
    )
    v3, v4 = v3.cpu().detach().numpy(), v4.cpu().detach().numpy()

    myplot2 = mp.plot(v, f, shading={"wireframe": False}, return_plot=True)
    myplot2.add_lines(v + v3 * avg, v - v3 * avg, shading={"line_color": "red"})
    myplot2.add_lines(v + v4 * avg, v - v4 * avg, shading={"line_color": "green"})

    # Compute GFrames from LBO eigenfunctions ##########################################
    g1, g2, gn = compute_gframes(
        verts=torch.tensor(v),
        faces=torch.tensor(f),
        normals=torch.tensor(mesh.vertex_normals),
        scalar_field=torch.tensor(h2),
    )
    g1, g2 = (
        g1.cpu().detach().numpy(),
        g2.cpu().detach().numpy(),
    )
    myplot3 = mp.plot(v, f, shading={"wireframe": False}, return_plot=True)
    myplot3.add_lines(v + g1 * avg, v - g1 * avg, shading={"line_color": "red"})
    myplot3.add_lines(v + g2 * avg, v - g2 * avg, shading={"line_color": "green"})
