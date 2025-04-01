import math
import trimesh
import torch
import igl
import scipy.sparse

import numpy as np
import robust_laplacian

from typing import Optional, Tuple
from torch_geometric.utils import add_self_loops, scatter, to_undirected


def load_mesh(
    file_path: str,
    show: bool = False,
    merge_tex: bool = True,
    bake_vert_colors: bool = False,
) -> trimesh.Trimesh:
    scene = trimesh.load(file_path, process=False)

    if hasattr(scene, "graph"):
        geometries = []
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]
            # get a copy of the geometry
            current = scene.geometry[geometry_name].copy()
            if isinstance(current, trimesh.Trimesh):
                # move the geometry vertices into the requested frame
                try:
                    current.apply_transform(transform)
                except RuntimeWarning:
                    print(f"troubles with {file_path}")

                # If there are pre-existing uvs in regions with a uniform colour
                # and no texture the visual concatenation fails.
                # Delete those uvs!
                try:
                    if current.visual.material.baseColorTexture is None:
                        current.visual.uv = None
                except AttributeError:
                    if current.visual.material.image is None:
                        current.visual.uv = None

                # save to our list of meshes
                geometries.append(current)

        if len(geometries) > 1:
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = geometries[0]
    else:
        mesh = scene

    trimesh.grouping.merge_vertices(mesh, merge_tex=merge_tex, merge_norm=True)

    if bake_vert_colors:
        mesh.visual = mesh.visual.to_color()

    if show:
        mesh.show()
    return mesh


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def sparse_torch_to_np(
    mat: torch.sparse.FloatTensor,
) -> scipy.sparse.csc_matrix:
    if len(mat.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(mat.shape))
    mat = mat.coalesce()
    indices = to_np(mat.indices())
    values = to_np(mat.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=mat.shape).tocsc()

    return mat


def stiefel_projx(x: torch.Tensor, driver: Optional[str] = None) -> torch.Tensor:
    assert driver is None or driver in ["gesvd", "gesvda", "gesvdj"]
    U, _, V = torch.linalg.svd(x, full_matrices=False, driver=driver)
    return torch.einsum("...ik,...kj->...ij", U, V)


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

        np_faces_t = face.numpy().T
        pd1, pd2, _, _ = igl.principal_curvature(pos.numpy(), np_faces_t)
        fpd1 = torch.tensor(igl.average_onto_faces(np_faces_t, pd1))
        fpd2 = torch.tensor(igl.average_onto_faces(np_faces_t, pd2))
        f_reference = torch.stack([fpd1, fpd2, face_normals], dim=2)
        f_reference_t = torch.transpose(f_reference, 1, 2)
        an_scale_mat = torch.diag(torch.tensor([1 / (1 + anisotropy), 1.0, 1.0])).to(
            torch.float64
        )
        scales_mat = torch.matmul(
            torch.matmul(f_reference, an_scale_mat), f_reference_t
        )
        angle = torch.tensor(rotation_angle)
        rotation_arount_normal_mat = torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle), 0],
                [torch.sin(angle), torch.cos(angle), 0],
                [0, 0, 1],
            ]
        ).to(torch.float64)
        anisotropy_mat = torch.matmul(
            torch.matmul(rotation_arount_normal_mat, scales_mat),
            rotation_arount_normal_mat.t(),
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
        area_deg.numpy(),
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
                lapl_eigsh, k=k_eig, M=mass_mat, sigma=eigs_sigma
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


def compute_tot_area(pos, faces):
    side_1 = pos[faces[1]] - pos[faces[0]]
    side_2 = pos[faces[2]] - pos[faces[0]]
    return side_1.cross(side_2).norm(p=2, dim=1).abs().sum() / 2


def to_basis(
    values: torch.Tensor, basis: torch.Tensor, massvec: torch.Tensor
) -> torch.Tensor:
    """
    Transform data in to an orthonormal basis (where orthonormal
    is wrt to massvec)
    Inputs:
      - values: (B,V,D)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,K,D) transformed values
    """
    basisT = basis.transpose(-2, -1)
    return torch.matmul(basisT, values * massvec.unsqueeze(-1))


def from_basis(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    Transform data out of an orthonormal basis
    Inputs:
      - values: (B,K,D)
      - basis: (B,V,K)
    Outputs:
      - (B,V,D) reconstructed values
    """
    if values.is_complex() or basis.is_complex():
        raise ValueError
    return torch.matmul(basis, values)


def heat_diffusion(
    x: torch.Tensor,
    mass: torch.Tensor,
    evals: torch.Tensor,
    evecs: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    # Transform to spectral
    x_spec = to_basis(x, evecs, mass)

    # Diffuse
    diffusion_coefs = torch.exp(-evals * time.unsqueeze(-1)).unsqueeze(-1)
    x_diffuse_spec = diffusion_coefs * x_spec

    # Transform back to per-vertex
    x_diffuse = from_basis(x_diffuse_spec, evecs)

    return x_diffuse


def cross(vec_1: torch.Tensor, vec_2: torch.Tensor) -> torch.Tensor:
    return torch.cross(vec_1, vec_2, dim=-1)


def dot(vec_1: torch.Tensor, vec_2: torch.Tensor) -> torch.Tensor:
    return torch.sum(vec_1 * vec_2, dim=-1)


def norm2(x: torch.Tensor) -> torch.Tensor:
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape)
    after norm along last dimension
    """
    return dot(x, x)


def farthest_point_sampling(points: torch.Tensor, n_sample: int) -> torch.Tensor:
    # Torch in, torch out. Returns a |V| mask with n_sample elements set to true

    N = points.shape[0]
    if n_sample > N:
        raise ValueError("not enough points to sample")

    chosen_mask = torch.zeros(N, dtype=torch.bool, device=points.device)
    min_dists = torch.ones(N, dtype=points.dtype, device=points.device) * float("inf")

    # pick the centermost first point
    # points = normalize_positions(points)  # they should be already centered
    i = torch.min(norm2(points), dim=0).indices
    chosen_mask[i] = True

    for _ in range(n_sample - 1):
        # update distance
        dists = norm2(points[i, :].unsqueeze(0) - points)
        min_dists = torch.minimum(dists, min_dists)

        # take the farthest
        i = torch.max(min_dists, dim=0).indices.item()
        chosen_mask[i] = True

    return chosen_mask


@torch.compile
def heat_diffusion_reduce(
    x: torch.Tensor,
    mass: torch.Tensor,
    evals: torch.Tensor,
    evecs: torch.Tensor,
    time: torch.Tensor,
):
    # Transform to spectral
    x_spec = to_basis(x, evecs, mass)

    # Diffuse
    diffusion_coefs = torch.exp(-evals * time.unsqueeze(-1)).unsqueeze(-1)
    x_diffuse_spec = diffusion_coefs * x_spec

    # Transform back to per-vertex
    x_diffuse = from_basis(x_diffuse_spec, evecs)

    # reduce
    return x_diffuse.sum(dim=0)


def big_trimesh_pcl(points, colours=None, radius=0.015):
    if isinstance(points, torch.Tensor):
        points = to_np(points)
    if isinstance(colours, torch.Tensor):
        colours = to_np(colours)
    pcl = [trimesh.creation.uv_sphere(radius=radius) for i in range(points.shape[0])]

    for i, p in enumerate(pcl):
        p.apply_translation(points[i])
        if colours is not None:
            p.visual.vertex_colors = np.zeros_like(p.vertices) + colours[i]
        else:
            p.visual.vertex_colors = np.zeros_like(p.vertices) + np.array([255, 0, 0])
    return pcl


if __name__ == "__main__":
    # mesh = load_mesh("objects/mech_drone.glb", show=False)
    mesh = load_mesh("../objects/spot_triangulated.obj", show=False)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    fnorm = np.array(mesh.face_normals)
    # a, b = compute_mesh_laplacian(verts, faces)
    # c, d = get_mesh_laplacian(torch.tensor(verts), torch.tensor(faces).T)

    a = 100
    r = math.radians(0)
    print(a)
    # lapl, massvec = compute_mesh_laplacian(verts, faces)

    lapl, mass = get_anisotropic_lbo(
        torch.tensor(verts),
        torch.tensor(faces).T,
        torch.tensor(fnorm),
        rotation_angle=r,
        anisotropy=a,
    )

    eval, evecs = compute_eig_laplacian(lapl=lapl, massvec=mass, k_eig=256)

    eval = torch.tensor(eval).unsqueeze(0).contiguous()
    evecs = torch.tensor(evecs).unsqueeze(0).contiguous()
    mass = torch.tensor(mass).unsqueeze(0).contiguous()

    colours = torch.ones_like(torch.tensor(mesh.vertices))
    colours = colours.unsqueeze(0)

    # c[10, :] = np.array([255, 0, 0])

    # for _ in range(1000):
    c_i = torch.zeros_like(colours)
    # c_i[:, torch.randint(c_i.shape[1] - 1, (1,)).item(), :] = torch.rand(
    #     3
    # ).unsqueeze(0)
    # c_i[torch.randint(c_i.shape[0] - 1, (1,)).item(), :] = torch.tensor(
    #     [1.0, 0, 0]
    # ).unsqueeze(0)
    c_i[:, 0, :] = torch.tensor([1.0, 0, 0]).unsqueeze(0)
    c_i = heat_diffusion(c_i, mass, eval, evecs, torch.tensor(0.1))
    colours += c_i

    # colours = colours * -1 + 1
    colours = (colours - colours.min()) / (colours.max() - colours.min())
    colours *= 255
    colours = colours.squeeze().numpy()

    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colours)
