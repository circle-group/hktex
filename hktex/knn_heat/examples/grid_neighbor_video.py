from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
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
        description="Render neighbor-selection videos for knn4 and bilinear4."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/interpolation/videos"),
        help="Output folder for videos.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=360,
        help="Number of frames in each video (360 at 30 FPS -> 12s).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Frame render DPI.",
    )
    return parser.parse_args()


def to_xy(angles: np.ndarray, anisotropies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.log1p(anisotropies)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return x, y


def build_query_path(num_frames: int, u_min: float, u_max: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    angles = np.pi * t
    u = u_min + (u_max - u_min) * (0.5 + 0.45 * np.sin(4.0 * np.pi * t))
    scales = np.expm1(u)
    return angles, scales


def select_neighbors(
    select_impl: str,
    rot_grid: torch.Tensor,
    scale_grid: torch.Tensor,
    angles: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gather = SpectralKnnGather(
        SpectralKnnGatherConfig(
            select_impl=select_impl,
            compile_select=False,
            gather_impl="full",
            compile_gather=False,
        )
    )
    angles_t = torch.tensor(angles, dtype=torch.float32)
    scales_t = torch.tensor(scales, dtype=torch.float32)
    idx, w = gather.select_grid(
        rot_grid=rot_grid,
        scale_grid=scale_grid,
        angles=angles_t,
        scales=scales_t,
        abs_sin=True,
        scale_map="log1p",
        grid_layout="rot_major",
        clamp_unit=True,
    )
    return idx.cpu().numpy(), w.cpu().numpy()


def draw_background(
    ax: plt.Axes,
    node_x: np.ndarray,
    node_y: np.ndarray,
    ring_inner: float,
    ring_outer: float,
    lim: float,
) -> None:
    ax.set_aspect("equal")
    ax.set_facecolor("#ffffff")
    ax.grid(alpha=0.18)

    for r in [ring_inner, ring_outer]:
        arc = Arc(
            (0.0, 0.0),
            width=2.0 * r,
            height=2.0 * r,
            theta1=0.0,
            theta2=180.0,
            linewidth=1.6,
            color="black",
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(arc)

    spokes = np.deg2rad(np.arange(0.0, 181.0, 30.0))
    for theta in spokes:
        x = ring_outer * math.cos(theta)
        y = ring_outer * math.sin(theta)
        ax.plot([0.0, x], [0.0, y], color="#b8c2cc", linewidth=0.8, alpha=0.7, zorder=1)

    ax.scatter(
        node_x,
        node_y,
        s=24,
        c="#f0f2f5",
        edgecolors="#455a64",
        linewidths=0.6,
        zorder=3,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0.0, lim)
    ax.set_xlabel(r"$x$ ($\log(1+\alpha)$-scaled)")
    ax.set_ylabel(r"$y$ ($\log(1+\alpha)$-scaled)")


def render_video(
    out_path: Path,
    select_impl: str,
    node_x: np.ndarray,
    node_y: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    idx: np.ndarray,
    w: np.ndarray,
    fps: int,
    dpi: int,
    ring_inner: float,
    ring_outer: float,
    lim: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"frames_{select_impl}_", dir=str(out_path.parent)) as tmp:
        tmp_dir = Path(tmp)
        plt.style.use("seaborn-v0_8-whitegrid")

        for i in range(qx.shape[0]):
            fig, ax = plt.subplots(figsize=(9, 6), facecolor="#f7f8fa")
            draw_background(ax, node_x, node_y, ring_inner, ring_outer, lim)
            method_label = "KNN-4" if select_impl == "knn4" else "Bilinear-4"
            ax.set_title(
                f"4-Neighbor Selection Along Query Trajectory ({method_label})",
                fontsize=13,
            )
            ax.text(
                0.015,
                0.975,
                f"frame {i+1}/{qx.shape[0]}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="#334155",
            )

            nidx = idx[i]
            nw = w[i]
            nx = node_x[nidx]
            ny = node_y[nidx]

            # Query point
            ax.scatter([qx[i]], [qy[i]], s=70, c="black", edgecolors="white", linewidths=0.8, zorder=6)

            # Connections and highlighted neighbors
            for j in range(4):
                ax.plot([qx[i], nx[j]], [qy[i], ny[j]], color="#0f172a", alpha=0.35, linewidth=1.0, zorder=4)
            sc = ax.scatter(
                nx,
                ny,
                s=120,
                c=nw,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                edgecolors="black",
                linewidths=1.1,
                zorder=7,
            )

            for j in range(4):
                ax.text(
                    nx[j] + 0.08,
                    ny[j] + 0.03,
                    f"w{j+1}={nw[j]:.2f}",
                    fontsize=8,
                    color="#0f172a",
                    zorder=8,
                )

            cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
            cbar.set_label("Neighbor weight")

            frame_path = tmp_dir / f"frame_{i:04d}.png"
            fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%04d.png"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    anisotropies = np.array([1.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0], dtype=np.float32)
    angles_deg = np.array(list(range(0, 181, 30)), dtype=np.float32)
    angles_rad = np.deg2rad(angles_deg)

    rot_grid = torch.tensor(angles_rad, dtype=torch.float32)
    scale_grid = torch.tensor(anisotropies, dtype=torch.float32)

    rr, aa = torch.meshgrid(rot_grid, scale_grid, indexing="ij")
    grid_angles = rr.reshape(-1).cpu().numpy()
    grid_anis = aa.reshape(-1).cpu().numpy()
    node_x, node_y = to_xy(grid_angles, grid_anis)

    ring_inner = float(np.log1p(1.0))
    ring_outer = float(np.log1p(200.0))
    lim = float(ring_outer * 1.08)

    q_angles, q_scales = build_query_path(args.frames, ring_inner, ring_outer)
    qx, qy = to_xy(q_angles, q_scales)

    idx_knn4, w_knn4 = select_neighbors("knn4", rot_grid, scale_grid, q_angles, q_scales)
    idx_bilin, w_bilin = select_neighbors("bilinear4", rot_grid, scale_grid, q_angles, q_scales)

    out_knn4 = args.out_dir / "neighbors_knn4.mp4"
    out_bilin = args.out_dir / "neighbors_bilinear4.mp4"

    render_video(
        out_knn4,
        "knn4",
        node_x,
        node_y,
        qx,
        qy,
        idx_knn4,
        w_knn4,
        args.fps,
        args.dpi,
        ring_inner,
        ring_outer,
        lim,
    )
    render_video(
        out_bilin,
        "bilinear4",
        node_x,
        node_y,
        qx,
        qy,
        idx_bilin,
        w_bilin,
        args.fps,
        args.dpi,
        ring_inner,
        ring_outer,
        lim,
    )

    print(f"Saved: {out_knn4}")
    print(f"Saved: {out_bilin}")


if __name__ == "__main__":
    main()
