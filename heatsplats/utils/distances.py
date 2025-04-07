import torch

__all__ = ["compute_biharmonic_distance"]


def compute_biharmonic_distance(
    evecs_i: torch.Tensor, evecs_j, evals: torch.Tensor, pairwise: bool = True
):
    """Compute the biharmonic distance between two sets of points on the surface given
    their eigenfunctions. Use eigenvectors and eigenvalues from the isotropic LBO!
    The eigenvalues are normally computed on the vertices of a mesh, their value can be
    gathered as 'evecs[idxs_i]'. If need to get eigenvalues of points belonging to faces,
    need to interpolate first. See example below.


    Args:
        evecs_i (torch.Tensor): [#I_pts, K] eigenvectors at points i
        evecs_j (torch.Tensor): [#J_pts, K] eigenvectors at points j
        evals (torch.Tensor): [K,] eigenvalues
        pairwise (bool, optional): whether it should compute all distances between
            I and J points. If False and #I_pts == #J_pts, it will compute the
            per attribute distances. Defaults to True.

    Returns:
        torch.Tensor:[#I_pts, #J_pts] distances between points i and j

    Example:
        >>> lapl, mass = utils.compute_mesh_laplacian(verts, faces)
        >>> evals, evecs = utils.compute_eig_laplacian(lapl, mass, 256)
        >>> v, f = torch.tensor(verts), torch.tensor(faces)
        >>> evals, evecs = torch.tensor(evals), torch.tensor(evecs)

        >>> # select eigenvalies of 2 points on faces of the mesh
        >>> fid = torch.tensor([0, 399])
        >>> bc = torch.tensor([[0.7410, 0.1356, 0.1234],
                               [0.3304, 0.1547, 0.5149]], dtype=torch.float64)

        >>> fevecs = utils.interpolate_barycentric_coords(f, fid, bc, evecs)

        >>> # select eigenvalies of 3 points on vertices of the mesh
        >>> pevecs = evecs[torch.tensor([10, 99, 800])]

        >>> # compute bharmonic distances
        >>> dists = utils.compute_biharmonic_distance(fevecs, pevecs, evals, True)
    """
    if evecs_i.shape[0] == evecs_j.shape[0] and not pairwise:
        diff = evecs_i - evecs_j
    else:
        diff = evecs_i.unsqueeze(0) - evecs_j.unsqueeze(1)
    return (evals.pow(-2) * diff.pow(2)).sum(dim=-1).sqrt()
