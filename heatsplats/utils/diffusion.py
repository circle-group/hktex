import torch

from .typing import *

__all__ = ["heat_diffusion", "heat_diffusion_reduce"]


def to_basis_at_points(
    values: Float[Tensor, "B P D"],
    basis: Float[Tensor, "B P K"],
    massvec: Float[Tensor, "B P"],
) -> Float[Tensor, "B P D"]:
    """
    Project a signal from non-uniform samples using quadrature weights.

    Inputs:
        - x_samples: The initial signal values at N points.
        - evecs_interp: The basis eigenvectors evaluated at the same N points.
        - quadrature_weights: The area weight for each sample point.

    Outputs:
        - (B, K, D) spectral coefficients.
    """
    # Weight the signal at each sample point by its area contribution.
    x_weighted = values * massvec.unsqueeze(-1)

    # Project the weighted signal onto the interpolated basis.
    # Sum( (w_i * x_i) * v_k(p_i) ) for each k.
    # (B, D, P) @ (B, P, K) -> (B, D, K)
    x_spec = torch.matmul(x_weighted.transpose(-2, -1), basis)

    # Transpose back to (B, K, D) convention.
    return x_spec.transpose(-2, -1)


def to_basis(
    values: Float[Tensor, "B V D"],
    basis: Float[Tensor, "B V K"],
    massvec: Float[Tensor, "B V"],
) -> Float[Tensor, "B K D"]:
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


def from_basis(
    values: Float[Tensor, "B K D"], basis: Float[Tensor, "B V K"]
) -> Float[Tensor, "B V D"]:
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
    x: Float[Tensor, "B V D"],
    mass: Float[Tensor, "B V"],
    evals: Float[Tensor, "B V"],
    evecs: Float[Tensor, "B V K"],
    time: Float[Tensor, "B"],
    weights_post_diff: Union[None, Float[Tensor, "B V"]] = None,
    at_vertices: bool = True,
) -> Float[Tensor, "B V D"]:
    # Transform to spectral
    if at_vertices:
        x_spec = to_basis(x, evecs, mass)
    else:
        x_spec = to_basis_at_points(x, evecs, mass)

    # Diffuse
    diffusion_coefs = torch.exp(-evals * time.unsqueeze(-1)).unsqueeze(-1)
    x_diffuse_spec = diffusion_coefs * x_spec

    # Transform back to per-vertex
    x_diffuse = from_basis(x_diffuse_spec, evecs)

    if weights_post_diff is not None:
        x_diffuse = x_diffuse * weights_post_diff.unsqueeze(-1)

    return x_diffuse


@torch.compile
def heat_diffusion_reduce(
    x: Float[Tensor, "B V D"],
    mass: Float[Tensor, "B V"],
    evals: Float[Tensor, "B V"],
    evecs: Float[Tensor, "B V K"],
    time: Float[Tensor, "B"],
    weights_post_diff: Union[None, Float[Tensor, "B V"]] = None,
) -> Float[Tensor, "B V D"]:
    # Transform to spectral
    x_spec = to_basis(x, evecs, mass)

    # Diffuse
    diffusion_coefs = torch.exp(-evals * time.unsqueeze(-1)).unsqueeze(-1)
    x_diffuse_spec = diffusion_coefs * x_spec

    # Transform back to per-vertex
    x_diffuse = from_basis(x_diffuse_spec, evecs)

    if weights_post_diff is not None:
        x_diffuse = x_diffuse * weights_post_diff.unsqueeze(-1)

    # reduce
    return x_diffuse.sum(dim=0)
