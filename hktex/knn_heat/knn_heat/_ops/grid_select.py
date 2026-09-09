from __future__ import annotations

from typing import Literal, Tuple

import torch
from jaxtyping import Float, Int64
from torch import Tensor

__all__ = [
    "_map_scales_1d",
    "_make_angle_scale_cartesian",
    "_make_grid_coords_cartesian",
    "_select_grid_knn4_from_axes",
    "_select_grid_cell4_bilinear",
]


def _map_scales_1d(
    scales: Float[Tensor, "N"],
    *,
    scale_map: Literal["log1p", "identity"] = "log1p",
) -> Float[Tensor, "N"]:
    if scale_map == "log1p":
        return torch.log1p(scales)
    if scale_map == "identity":
        return scales
    raise ValueError(f"Unsupported scale_map: {scale_map}")


def _make_angle_scale_cartesian(
    angles: Float[Tensor, "N"],
    scales: Float[Tensor, "N"],
    *,
    abs_sin: bool = True,
    scale_map: Literal["log1p", "identity"] = "log1p",
) -> Float[Tensor, "N 2"]:
    n = angles.shape[0]
    assert scales.shape[0] == n

    scales_mapped = _map_scales_1d(scales, scale_map=scale_map)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    if abs_sin:
        sin_a = sin_a.abs()

    return torch.stack([cos_a * scales_mapped, sin_a * scales_mapped], dim=1)


def _make_grid_coords_cartesian(
    rot_grid: Float[Tensor, "R"],
    scale_grid: Float[Tensor, "A"],
    *,
    abs_sin: bool = True,
    scale_map: Literal["log1p", "identity"] = "log1p",
    grid_layout: Literal["rot_major", "scale_major"] = "rot_major",
) -> Float[Tensor, "M 2"]:
    rr, aa = torch.meshgrid(rot_grid, scale_grid, indexing="ij")  # [R, A], [R, A]

    if grid_layout == "rot_major":
        rr = rr.reshape(-1)
        aa = aa.reshape(-1)
    elif grid_layout == "scale_major":
        rr = rr.transpose(0, 1).reshape(-1)
        aa = aa.transpose(0, 1).reshape(-1)
    else:
        raise ValueError(f"Unsupported grid_layout: {grid_layout}")

    return _make_angle_scale_cartesian(
        rr,
        aa,
        abs_sin=abs_sin,
        scale_map=scale_map,
    )


def _select_grid_knn4_from_axes(
    rot_grid: Float[Tensor, "R"],
    scale_grid: Float[Tensor, "A"],
    angles: Float[Tensor, "S"],
    scales: Float[Tensor, "S"],
    *,
    abs_sin: bool = True,
    scale_map: Literal["log1p", "identity"] = "log1p",
    grid_layout: Literal["rot_major", "scale_major"] = "rot_major",
    eps: float = 1e-10,
    clamp_unit: bool = True,  # unused
) -> Tuple[Int64[Tensor, "S 4"], Float[Tensor, "S 4"]]:
    # rot_grid: [R], scale_grid: [A], angles: [S], scales: [S]
    # returns: m_neigh_idx [S, 4], m_neigh_w [S, 4]
    s = angles.shape[0]

    grid_coords_cartesian = _make_grid_coords_cartesian(
        rot_grid,
        scale_grid,
        abs_sin=abs_sin,
        scale_map=scale_map,
        grid_layout=grid_layout,
    )  # [M, 2]
    query_coords_cartesian = _make_angle_scale_cartesian(
        angles,
        scales,
        abs_sin=abs_sin,
        scale_map=scale_map,
    )  # [S, 2]

    diff = grid_coords_cartesian.unsqueeze(1) - query_coords_cartesian.unsqueeze(0)
    dist2 = (diff * diff).sum(dim=-1)  # [M, S]

    d2, idx = dist2.topk(k=4, largest=False, dim=0)  # [4, S], [4, S]
    neigh_idx = idx.t().contiguous()  # [S, 4]
    d2 = d2.t().contiguous()  # [S, 4]

    neigh_w = 1.0 / torch.clamp(d2, min=eps)
    neigh_w = neigh_w / neigh_w.sum(dim=1, keepdim=True)

    if neigh_idx.dtype != torch.int64:
        neigh_idx = neigh_idx.to(torch.int64)

    return neigh_idx, neigh_w


def _select_grid_cell4_bilinear(
    rot_grid: Float[Tensor, "R"],
    scale_grid: Float[Tensor, "A"],
    angles: Float[Tensor, "S"],
    scales: Float[Tensor, "S"],
    *,
    abs_sin: bool = True,  # unused
    scale_map: Literal["log1p", "identity"] = "log1p",
    grid_layout: Literal["rot_major", "scale_major"] = "rot_major",
    eps: float = 1e-10,
    clamp_unit: bool = True,
) -> Tuple[Int64[Tensor, "S 4"], Float[Tensor, "S 4"]]:
    # rot_grid: [R] sorted, scale_grid: [A] sorted
    # returns: m_neigh_idx [S, 4], m_neigh_w [S, 4]
    u_grid = _map_scales_1d(scale_grid, scale_map=scale_map)  # [A]
    u = _map_scales_1d(scales, scale_map=scale_map)  # [S]

    r1 = torch.searchsorted(rot_grid, angles, right=False)
    a1 = torch.searchsorted(u_grid, u, right=False)

    r1 = torch.clamp(r1, min=1, max=rot_grid.shape[0] - 1).to(torch.int64)
    a1 = torch.clamp(a1, min=1, max=u_grid.shape[0] - 1).to(torch.int64)

    r0 = r1 - 1
    a0 = a1 - 1

    ang0 = rot_grid[r0]
    ang1 = rot_grid[r1]
    u0 = u_grid[a0]
    u1 = u_grid[a1]

    t = (angles - ang0) / torch.clamp(ang1 - ang0, min=eps)
    s = (u - u0) / torch.clamp(u1 - u0, min=eps)
    if clamp_unit:
        t = t.clamp(0.0, 1.0)
        s = s.clamp(0.0, 1.0)

    # Corner indices from cartesian product of {r0, r1} x {a0, a1}
    r_pair = torch.stack([r0, r1], dim=1)  # [S, 2]
    a_pair = torch.stack([a0, a1], dim=1)  # [S, 2]

    r_corner = r_pair.unsqueeze(2)  # [S, 2, 1]
    a_corner = a_pair.unsqueeze(1)  # [S, 1, 2]

    n_rot = int(rot_grid.shape[0])
    n_scale = int(scale_grid.shape[0])

    if grid_layout == "rot_major":
        idx = r_corner * n_scale + a_corner  # [S, 2, 2]
    elif grid_layout == "scale_major":
        idx = a_corner * n_rot + r_corner  # [S, 2, 2]
    else:
        raise ValueError(f"Unsupported grid_layout: {grid_layout}")

    # Weights are outer product of [1-t, t] and [1-s, s]
    w_r = torch.stack([1.0 - t, t], dim=1)  # [S, 2]
    w_a = torch.stack([1.0 - s, s], dim=1)  # [S, 2]
    w = w_r.unsqueeze(2) * w_a.unsqueeze(1)  # [S, 2, 2]

    # Match the corner order used by the reshape:
    # [S, 2, 2] reshapes with last dim fastest:
    # (r0,a0), (r0,a1), (r1,a0), (r1,a1)
    neigh_idx = idx.reshape(idx.shape[0], 4).contiguous().to(torch.int64)
    neigh_w = w.reshape(w.shape[0], 4).contiguous()

    return neigh_idx, neigh_w
