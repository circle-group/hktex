from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Arc

from knn_heat.config import SpectralKnnGatherConfig
from knn_heat.knn_gather import SpectralKnnGather


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize anisotropy-angle grid and interpolation fields."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/interpolation"),
        help="Output folder for all generated figures.",
    )
    return parser.parse_args()


def target_fn(angles: torch.Tensor, anisotropies: torch.Tensor) -> torch.Tensor:
    """Smooth, nontrivial scalar field over (angle, anisotropy)."""
    u = torch.log1p(anisotropies)
    t1 = 0.9 * torch.cos(2.0 * angles) * torch.exp(-0.22 * u)
    t2 = 0.35 * torch.sin(4.0 * angles) * (u / (u + 1.2))
    t3 = 0.2 * torch.cos(7.0 * angles) / (1.0 + 0.18 * u)
    return t1 + t2 + t3


def to_xy(angles: np.ndarray, anisotropies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.log1p(anisotropies)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return x, y


def save_figure_with_pdf(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")


def draw_base_grid(
    out_path: Path,
    anisotropies: np.ndarray,
    angles_rad: np.ndarray,
    lim: float,
) -> None:
    anis_r = np.log1p(anisotropies)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#f7f8fa")
    ax.set_aspect("equal")
    ax.set_title(
        "Log-Scaled Anisotropy-Angle Grid (Upper Semicircle)",
        fontsize=14,
        pad=12,
    )
    ax.set_facecolor("#ffffff")

    for r in anis_r:
        arc = Arc(
            (0.0, 0.0),
            width=2.0 * r,
            height=2.0 * r,
            theta1=0.0,
            theta2=180.0,
            linewidth=1.2,
            alpha=0.35,
            color="#7a8ca3",
        )
        ax.add_patch(arc)

    r_max = float(anis_r.max())
    for theta in angles_rad:
        x = r_max * math.cos(theta)
        y = r_max * math.sin(theta)
        ax.plot([0.0, x], [0.0, y], color="#a7b3c2", alpha=0.6, linewidth=1.0)

    rr, tt = np.meshgrid(anis_r, angles_rad, indexing="ij")
    x = (rr * np.cos(tt)).reshape(-1)
    y = (rr * np.sin(tt)).reshape(-1)

    c = np.repeat(anisotropies, len(angles_rad))
    sc = ax.scatter(
        x,
        y,
        s=34,
        c=c,
        cmap="viridis",
        edgecolors="#1f2d3d",
        linewidths=0.3,
        alpha=0.95,
        zorder=3,
    )

    for a_true, a_r in zip(anisotropies, anis_r):
        ax.text(
            0.05,
            float(a_r),
            f"{int(a_true)}",
            fontsize=9,
            color="#2f3e4e",
            va="center",
        )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(0.0, lim)
    ax.set_xlabel(r"$x$ ($\log(1+\alpha)$-scaled)")
    ax.set_ylabel(r"$y$ ($\log(1+\alpha)$-scaled)")
    ax.grid(alpha=0.18)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Anisotropy (original scale)")

    save_figure_with_pdf(fig, out_path)
    plt.close(fig)


def draw_interpolation_figure(
    out_path: Path,
    method: str,
    xq: np.ndarray,
    yq: np.ndarray,
    pred: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_vals: np.ndarray,
    vmin: float,
    vmax: float,
    ring_inner: float,
    ring_outer: float,
    lim: float,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#f7f8fa")
    ax.set_aspect("equal")
    ax.set_facecolor("#ffffff")
    method_label = "KNN-4" if method == "knn4" else "Bilinear-4"
    ax.set_title(
        f"Interpolated Function Field ({method_label})",
        fontsize=14,
        pad=12,
    )

    im = ax.scatter(
        xq,
        yq,
        c=pred,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=10,
        alpha=0.95,
        linewidths=0.0,
        rasterized=True,
    )

    ax.scatter(
        grid_x,
        grid_y,
        c=grid_vals,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=46,
        edgecolors="#111827",
        linewidths=0.35,
        zorder=4,
    )

    # Boundaries for interpolation region in log1p radius:
    # inner boundary (log1p(1)) and outer boundary (log1p(200)).
    inner_arc = Arc(
        (0.0, 0.0),
        width=2.0 * ring_inner,
        height=2.0 * ring_inner,
        theta1=0.0,
        theta2=180.0,
        color="black",
        linewidth=1.6,
        alpha=0.95,
        zorder=5,
    )
    outer_arc = Arc(
        (0.0, 0.0),
        width=2.0 * ring_outer,
        height=2.0 * ring_outer,
        theta1=0.0,
        theta2=180.0,
        color="black",
        linewidth=1.6,
        alpha=0.95,
        zorder=5,
    )
    ax.add_patch(inner_arc)
    ax.add_patch(outer_arc)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(0.0, lim)
    ax.set_xlabel(r"$x$ ($\log(1+\alpha)$-scaled)")
    ax.set_ylabel(r"$y$ ($\log(1+\alpha)$-scaled)")
    ax.grid(alpha=0.18)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Function value")

    save_figure_with_pdf(fig, out_path)
    plt.close(fig)


def draw_direct_function_figure(
    out_path: Path,
    xq: np.ndarray,
    yq: np.ndarray,
    direct_vals: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_vals: np.ndarray,
    vmin: float,
    vmax: float,
    ring_inner: float,
    ring_outer: float,
    lim: float,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#f7f8fa")
    ax.set_aspect("equal")
    ax.set_facecolor("#ffffff")
    ax.set_title(
        "Ground-Truth Function Field (No Interpolation)",
        fontsize=14,
        pad=12,
    )

    im = ax.scatter(
        xq,
        yq,
        c=direct_vals,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=10,
        alpha=0.95,
        linewidths=0.0,
        rasterized=True,
    )

    ax.scatter(
        grid_x,
        grid_y,
        c=grid_vals,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=46,
        edgecolors="#111827",
        linewidths=0.35,
        zorder=4,
    )

    inner_arc = Arc(
        (0.0, 0.0),
        width=2.0 * ring_inner,
        height=2.0 * ring_inner,
        theta1=0.0,
        theta2=180.0,
        color="black",
        linewidth=1.6,
        alpha=0.95,
        zorder=5,
    )
    outer_arc = Arc(
        (0.0, 0.0),
        width=2.0 * ring_outer,
        height=2.0 * ring_outer,
        theta1=0.0,
        theta2=180.0,
        color="black",
        linewidth=1.6,
        alpha=0.95,
        zorder=5,
    )
    ax.add_patch(inner_arc)
    ax.add_patch(outer_arc)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(0.0, lim)
    ax.set_xlabel(r"$x$ ($\log(1+\alpha)$-scaled)")
    ax.set_ylabel(r"$y$ ($\log(1+\alpha)$-scaled)")
    ax.grid(alpha=0.18)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Function value")

    save_figure_with_pdf(fig, out_path)
    plt.close(fig)


def interpolate_with_select_impl(
    select_impl: str,
    rot_grid: torch.Tensor,
    scale_grid: torch.Tensor,
    grid_vals: torch.Tensor,
    query_angles: torch.Tensor,
    query_scales: torch.Tensor,
) -> torch.Tensor:
    gather = SpectralKnnGather(
        SpectralKnnGatherConfig(
            select_impl=select_impl,
            compile_select=False,
            gather_impl="full",
            compile_gather=False,
        )
    )

    idx, w = gather.select_grid(
        rot_grid=rot_grid,
        scale_grid=scale_grid,
        angles=query_angles,
        scales=query_scales,
        abs_sin=True,
        scale_map="log1p",
        grid_layout="rot_major",
        clamp_unit=True,
    )
    return (grid_vals[idx] * w).sum(dim=1)


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_grid = out_dir / "grid_layout.png"
    out_knn4 = out_dir / "interp_knn4.png"
    out_bilinear4 = out_dir / "interp_bilinear4.png"
    out_direct = out_dir / "function_direct.png"

    anisotropies = np.array([1.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0], dtype=np.float32)
    angles_deg = np.array(list(range(0, 181, 30)), dtype=np.float32)
    angles_rad = np.deg2rad(angles_deg)
    shared_lim = float(np.log1p(anisotropies.max()) * 1.08)

    draw_base_grid(out_grid, anisotropies, angles_rad, shared_lim)

    device = torch.device("cpu")
    rot_grid = torch.tensor(angles_rad, dtype=torch.float32, device=device)
    scale_grid = torch.tensor(anisotropies, dtype=torch.float32, device=device)

    # Grid node values in rot-major flatten order used by selector.
    rr, aa = torch.meshgrid(rot_grid, scale_grid, indexing="ij")
    grid_vals = target_fn(rr.reshape(-1), aa.reshape(-1))

    # Dense query domain over upper semicircle for smooth visualization.
    q_angles = torch.linspace(0.0, float(torch.pi), steps=361, dtype=torch.float32, device=device)
    q_log_scales = torch.linspace(
        0.0,
        float(np.log1p(anisotropies.max())),
        steps=300,
        dtype=torch.float32,
        device=device,
    )
    q_scales = torch.expm1(q_log_scales)
    qq_theta, qq_scale = torch.meshgrid(q_angles, q_scales, indexing="ij")
    q_theta_flat = qq_theta.reshape(-1)
    q_scale_flat = qq_scale.reshape(-1)

    pred_knn4 = interpolate_with_select_impl(
        "knn4", rot_grid, scale_grid, grid_vals, q_theta_flat, q_scale_flat
    )
    pred_bilinear4 = interpolate_with_select_impl(
        "bilinear4", rot_grid, scale_grid, grid_vals, q_theta_flat, q_scale_flat
    )
    direct_vals = target_fn(q_theta_flat, q_scale_flat)

    qx, qy = to_xy(qq_theta.cpu().numpy(), qq_scale.cpu().numpy())
    qx_flat = qx.reshape(-1)
    qy_flat = qy.reshape(-1)

    grid_angles = rr.reshape(-1).cpu().numpy()
    grid_anis = aa.reshape(-1).cpu().numpy()
    gx, gy = to_xy(grid_angles, grid_anis)
    gx_flat = gx.reshape(-1)
    gy_flat = gy.reshape(-1)

    grid_vals_np = grid_vals.cpu().numpy()
    pred_knn4_np = pred_knn4.cpu().numpy()
    pred_bilinear4_np = pred_bilinear4.cpu().numpy()
    direct_vals_np = direct_vals.cpu().numpy()

    vmin = float(
        min(
            grid_vals_np.min(),
            pred_knn4_np.min(),
            pred_bilinear4_np.min(),
            direct_vals_np.min(),
        )
    )
    vmax = float(
        max(
            grid_vals_np.max(),
            pred_knn4_np.max(),
            pred_bilinear4_np.max(),
            direct_vals_np.max(),
        )
    )
    ring_inner = float(np.log1p(1.0))
    ring_outer = float(np.log1p(200.0))

    draw_interpolation_figure(
        out_knn4,
        "knn4",
        qx_flat,
        qy_flat,
        pred_knn4_np,
        gx_flat,
        gy_flat,
        grid_vals_np,
        vmin,
        vmax,
        ring_inner,
        ring_outer,
        shared_lim,
    )
    draw_interpolation_figure(
        out_bilinear4,
        "bilinear4",
        qx_flat,
        qy_flat,
        pred_bilinear4_np,
        gx_flat,
        gy_flat,
        grid_vals_np,
        vmin,
        vmax,
        ring_inner,
        ring_outer,
        shared_lim,
    )
    draw_direct_function_figure(
        out_direct,
        qx_flat,
        qy_flat,
        direct_vals_np,
        gx_flat,
        gy_flat,
        grid_vals_np,
        vmin,
        vmax,
        ring_inner,
        ring_outer,
        shared_lim,
    )
    print(f"Shared axis limits: x=[{-shared_lim:.6f}, {shared_lim:.6f}], y=[0.000000, {shared_lim:.6f}]")
    print(f"Shared color range (direct/interp): vmin={vmin:.6f}, vmax={vmax:.6f}")


if __name__ == "__main__":
    main()
