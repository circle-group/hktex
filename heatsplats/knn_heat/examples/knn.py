from __future__ import annotations

import argparse
import time

import torch

from knn_heat.config import SpectralKnnGatherConfig
from knn_heat.knn_gather import SpectralKnnGather


def _parse_bool(x: str) -> bool:
    v = x.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpectralKnnGather example.")
    parser.add_argument(
        "--select-impl",
        type=str,
        choices=["knn4", "bilinear4"],
        default="bilinear4",
        help="Grid neighbor selection implementation.",
    )
    parser.add_argument(
        "--compile-select",
        type=_parse_bool,
        default=False,
        help="Whether to torch.compile select op (true/false).",
    )
    parser.add_argument(
        "--gather-impl",
        type=str,
        choices=["full", "stream_vertices", "stream_r_i"],
        default="stream_vertices",
        help="Evec gather implementation.",
    )
    parser.add_argument(
        "--compile-gather",
        type=_parse_bool,
        default=False,
        help="Whether to torch.compile gather op (true/false).",
    )
    return parser.parse_args()


def _random_barycentric(n: int, device: torch.device) -> torch.Tensor:
    # Positive rows summing to 1.
    return torch.softmax(torch.randn(n, 3, device=device, dtype=torch.float32), dim=-1)


class _Timer:
    def __init__(self, device: torch.device) -> None:
        self._use_cuda = device.type == "cuda"
        self._cpu_t0: float | None = None
        self._cuda_start: torch.cuda.Event | None = None
        self._cuda_end: torch.cuda.Event | None = None

    def start(self) -> None:
        if self._use_cuda:
            self._cuda_start = torch.cuda.Event(enable_timing=True)
            self._cuda_start.record()
        else:
            self._cpu_t0 = time.perf_counter()

    def stop_ms(self) -> float:
        if self._use_cuda:
            assert self._cuda_start is not None
            self._cuda_end = torch.cuda.Event(enable_timing=True)
            self._cuda_end.record()
            torch.cuda.synchronize()
            return float(self._cuda_start.elapsed_time(self._cuda_end))
        assert self._cpu_t0 is not None
        t1 = time.perf_counter()
        return (t1 - self._cpu_t0) * 1e3


def main() -> None:
    args = _parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Problem sizes.
    s = 512  # number of sources
    q = 256  # number of queries
    k = 16  # KNN per query
    r = 16  # number of rotation grid points
    a = 8  # number of anisotropy grid points
    m = r * a  # number of spectral bins on grid
    v = 2048  # vertices per mesh
    e = 64  # spectral basis size

    cfg = SpectralKnnGatherConfig(
        select_impl=args.select_impl,
        compile_select=args.compile_select,
        gather_impl=args.gather_impl,
        compile_gather=args.compile_gather,
    )
    gather = SpectralKnnGather(cfg)
    print("config:", cfg)

    # Rotation/scale grid and source conditioning.
    rot_grid = torch.linspace(-torch.pi, torch.pi, steps=r, device=device)
    scale_grid = torch.linspace(0.1, 2.0, steps=a, device=device)
    src_angles = ((torch.rand(s, device=device) * 2.0 - 1.0) * torch.pi).requires_grad_(
        True
    )
    src_scales = (torch.rand(s, device=device) * 1.9 + 0.1).requires_grad_(True)

    # Spectral data treated as constants in the gather path.
    evals = torch.rand(m, e, device=device, dtype=torch.float32)
    evecs = torch.randn(m, v, e, device=device, dtype=torch.float32)

    # Source/query geometry.
    src_vert_idx = torch.randint(0, v, (s, 3), device=device, dtype=torch.int64)
    src_bary_coords = _random_barycentric(s, device).requires_grad_(True)
    query_vert_idx = torch.randint(0, v, (q, 3), device=device, dtype=torch.int64)
    query_bary_coords = _random_barycentric(q, device).requires_grad_(True)
    src_knn_idx = torch.randint(0, s, (q, k), device=device, dtype=torch.int64)

    # 1) Build per-source quantities and 2) query gather in loop so gradients from
    # query loss can flow to source conditioning each step.
    total_timer = _Timer(device)
    total_timer.start()
    gather_src_ms = 0.0
    gather_query_ms = 0.0
    n_repeats = 3
    for step in range(n_repeats):
        if src_angles.grad is not None:
            src_angles.grad = None
        if src_scales.grad is not None:
            src_scales.grad = None
        if src_bary_coords.grad is not None:
            src_bary_coords.grad = None
        if query_bary_coords.grad is not None:
            query_bary_coords.grad = None

        src_timer = _Timer(device)
        src_timer.start()
        m_neigh_idx, m_neigh_w = gather.select_grid(
            rot_grid=rot_grid,
            scale_grid=scale_grid,
            angles=src_angles,
            scales=src_scales,
            abs_sin=True,
            scale_map="log1p",
            grid_layout="rot_major",
            clamp_unit=True,
        )
        m_neigh_w.retain_grad()

        evals_src, evecs_src = gather.gather_src(
            m_neigh_idx=m_neigh_idx,
            m_neigh_w=m_neigh_w,
            src_vert_idx=src_vert_idx,
            src_bary_coords=src_bary_coords,
            evals=evals,
            evecs=evecs,
        )
        src_norm = evecs_src.norm(dim=-1) + 1e-6
        gather_src_ms += src_timer.stop_ms()

        query_timer = _Timer(device)
        query_timer.start()
        evals_knn, evecs_knn, norm_knn = gather.gather_queries(
            src_knn_idx=src_knn_idx,
            m_neigh_idx=m_neigh_idx,
            m_neigh_w=m_neigh_w,
            vert_idx=query_vert_idx,
            bary_coords=query_bary_coords,
            evals=evals,
            evecs=evecs,
            src_norm=src_norm,
        )

        # Dummy differentiable objective.
        loss = evals_knn.mean() + evecs_knn.square().mean()
        if norm_knn is not None:
            loss = loss + 0.01 * norm_knn.mean()
        loss.backward()
        gather_query_ms += query_timer.stop_ms()

        print(
            f"step {step}: "
            f"evals_knn={tuple(evals_knn.shape)} "
            f"evecs_knn={tuple(evecs_knn.shape)} "
            f"norm_knn={None if norm_knn is None else tuple(norm_knn.shape)} "
            f"grad_src_angles={src_angles.grad is not None} "
            f"grad_src_scales={src_scales.grad is not None} "
            f"grad_m_w={m_neigh_w.grad is not None} "
            f"grad_src_bary={src_bary_coords.grad is not None} "
            f"grad_query_bary={query_bary_coords.grad is not None}"
        )

    total_ms = total_timer.stop_ms()
    print(f"end-to-end x{n_repeats}: {total_ms:.2f} ms")
    print(f"avg gather_src per step: {gather_src_ms / n_repeats:.2f} ms")
    print(f"avg gather_queries per step: {gather_query_ms / n_repeats:.2f} ms")
    print("sample evals_src[0,:8]:", evals_src[0, :8].tolist())
    print("sample evecs_knn[0,0,:8]:", evecs_knn[0, 0, :8].tolist())


if __name__ == "__main__":
    main()
