from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch

from knn_heat import (
    FaissGpuFlatIndex,
    FaissGpuIndexConfig,
    KnnPostDiffWeight,
    KnnPostDiffWeightConfig,
    SpectralKnnGather,
    SpectralKnnGatherConfig,
    SpectralKnnHeat,
    SpectralKnnHeatConfig,
)


def _parse_bool(x: str) -> bool:
    v = x.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full pipeline: FAISS KNN -> spectral gather -> post-diff weights -> heat diffusion."
        )
    )
    parser.add_argument("--s", type=int, default=512, help="Number of sources.")
    parser.add_argument("--q", type=int, default=256, help="Number of queries.")
    parser.add_argument("--k", type=int, default=16, help="KNN per query.")
    parser.add_argument("--d", type=int, default=128, help="Embedding dim for FAISS KNN.")
    parser.add_argument("--e", type=int, default=64, help="Spectral dimension.")
    parser.add_argument("--v", type=int, default=2048, help="Vertices per mesh.")
    parser.add_argument("--r", type=int, default=16, help="Rotation grid size.")
    parser.add_argument("--a", type=int, default=8, help="Scale grid size.")
    parser.add_argument(
        "--weight-kernel",
        type=str,
        choices=["inverse", "gaussian"],
        default="inverse",
        help="Post-diff weighting kernel.",
    )
    parser.add_argument(
        "--weight-std",
        type=float,
        default=0.3,
        help="Gaussian std if --weight-kernel gaussian.",
    )
    parser.add_argument(
        "--compile-gather",
        type=_parse_bool,
        default=True,
        help="Whether to torch.compile gather internals.",
    )
    parser.add_argument(
        "--compile-dist",
        type=_parse_bool,
        default=True,
        help="Whether to torch.compile differentiable KNN distance recomputation.",
    )
    parser.add_argument(
        "--compile-heat",
        type=_parse_bool,
        default=True,
        help="Whether to torch.compile heat ops.",
    )
    parser.add_argument(
        "--compile-weight",
        type=_parse_bool,
        default=True,
        help="Whether to torch.compile post-weight kernel.",
    )
    parser.add_argument(
        "--query-steps",
        type=int,
        default=10,
        help="Number of query iterations in the timed pass.",
    )
    parser.add_argument(
        "--warmup-query-steps",
        type=int,
        default=2,
        help="Number of query iterations in warmup pass.",
    )
    return parser.parse_args()


def _random_barycentric(n: int, device: torch.device) -> torch.Tensor:
    return torch.softmax(torch.randn(n, 3, device=device, dtype=torch.float32), dim=-1)


def _event_start() -> torch.cuda.Event:
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def _event_elapsed_ms(start: torch.cuda.Event) -> float:
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _print_stats(name: str, values_ms: List[float]) -> None:
    t = torch.tensor(values_ms, dtype=torch.float64)
    mean = float(t.mean().item())
    std = float(t.std(unbiased=False).item())
    mn = float(t.min().item())
    mx = float(t.max().item())
    print(f"  {name:>14s}: mean={mean:8.3f} ms  std={std:7.3f}  min={mn:8.3f}  max={mx:8.3f}")


@dataclass
class PipelineState:
    device: torch.device
    s: int
    q: int
    k: int
    d: int
    e: int
    v: int
    m: int
    faiss_index: FaissGpuFlatIndex
    gather: SpectralKnnGather
    heat: SpectralKnnHeat
    weighter: KnnPostDiffWeight
    src_embed: torch.Tensor
    src_angles: torch.Tensor
    src_scales: torch.Tensor
    src_bary: torch.Tensor
    rot_grid: torch.Tensor
    scale_grid: torch.Tensor
    evals_grid: torch.Tensor
    evecs_grid: torch.Tensor
    src_vert_idx: torch.Tensor


def build_state(args: argparse.Namespace, device: torch.device) -> PipelineState:
    s, q, k = args.s, args.q, args.k
    d, e, v = args.d, args.e, args.v
    m = args.r * args.a

    src_embed = (
        torch.randn(s, d, device=device, dtype=torch.float32)
        .contiguous()
        .requires_grad_(True)
    )
    src_angles = ((torch.rand(s, device=device) * 2.0 - 1.0) * torch.pi).requires_grad_(True)
    src_scales = (torch.rand(s, device=device) * 1.9 + 0.1).requires_grad_(True)
    src_bary = _random_barycentric(s, device).requires_grad_(True)

    rot_grid = torch.linspace(-torch.pi, torch.pi, steps=args.r, device=device)
    scale_grid = torch.linspace(0.1, 2.0, steps=args.a, device=device)
    evals_grid = torch.rand(m, e, device=device, dtype=torch.float32)
    evecs_grid = torch.randn(m, v, e, device=device, dtype=torch.float32)
    src_vert_idx = torch.randint(0, v, (s, 3), device=device, dtype=torch.int64)

    faiss_cfg = FaissGpuIndexConfig(
        metric="l2",
        use_float16=False,
        compile_distances=args.compile_dist,
        distance_impl="bmm",
        build_db_norm=True,
    )
    gather_cfg = SpectralKnnGatherConfig(
        select_impl="bilinear4",
        compile_select=args.compile_gather,
        gather_impl="stream_vertices",
        compile_gather=args.compile_gather,
    )
    heat_cfg = SpectralKnnHeatConfig(compile_heat=args.compile_heat, eps=1e-8)
    weight_cfg = KnnPostDiffWeightConfig(
        kernel=args.weight_kernel,
        gaussian_std=args.weight_std,
        compile_kernel=args.compile_weight,
        normalize=True,
    )

    return PipelineState(
        device=device,
        s=s,
        q=q,
        k=k,
        d=d,
        e=e,
        v=v,
        m=m,
        faiss_index=FaissGpuFlatIndex(faiss_cfg),
        gather=SpectralKnnGather(gather_cfg),
        heat=SpectralKnnHeat(heat_cfg),
        weighter=KnnPostDiffWeight(weight_cfg),
        src_embed=src_embed,
        src_angles=src_angles,
        src_scales=src_scales,
        src_bary=src_bary,
        rot_grid=rot_grid,
        scale_grid=scale_grid,
        evals_grid=evals_grid,
        evecs_grid=evecs_grid,
        src_vert_idx=src_vert_idx,
    )


def run_pipeline_iteration(
    state: PipelineState,
    *,
    query_steps: int,
    record_timings: bool,
) -> tuple[
    torch.Tensor,
    Dict[str, List[float]],
    Dict[str, tuple[int, ...]],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
]:
    timings: Dict[str, List[float]] = {
        "faiss_build": [],
        "select_grid": [],
        "gather_src": [],
        "heat_build": [],
        "query_total": [],
        "query_search": [],
        "query_dist": [],
        "query_weights": [],
        "query_gather": [],
        "query_heat": [],
    }

    # Non-query stages (run once per iteration).
    t0 = _event_start()
    state.faiss_index.build(state.src_embed)
    ms = _event_elapsed_ms(t0)
    if record_timings:
        timings["faiss_build"].append(ms)

    t0 = _event_start()
    m_neigh_idx, m_neigh_w = state.gather.select_grid(
        rot_grid=state.rot_grid,
        scale_grid=state.scale_grid,
        angles=state.src_angles,
        scales=state.src_scales,
        abs_sin=True,
        scale_map="log1p",
        grid_layout="rot_major",
        clamp_unit=True,
    )
    ms = _event_elapsed_ms(t0)
    if record_timings:
        timings["select_grid"].append(ms)

    t0 = _event_start()
    evals_src, evecs_src = state.gather.gather_src(
        m_neigh_idx=m_neigh_idx,
        m_neigh_w=m_neigh_w,
        src_vert_idx=state.src_vert_idx,
        src_bary_coords=state.src_bary,
        evals=state.evals_grid,
        evecs=state.evecs_grid,
    )
    ms = _event_elapsed_ms(t0)
    if record_timings:
        timings["gather_src"].append(ms)

    t0 = _event_start()
    t_heat = torch.tensor(0.35, device=state.device, dtype=torch.float32)
    state.heat.build(evals_src=evals_src, evecs_src=evecs_src, time=t_heat)
    ms = _event_elapsed_ms(t0)
    if record_timings:
        timings["heat_build"].append(ms)

    loss = torch.zeros((), device=state.device, dtype=torch.float32)
    last_shapes: Dict[str, tuple[int, ...]] = {}
    query_embeds: List[torch.Tensor] = []
    query_barys: List[torch.Tensor] = []
    weights_posts: List[torch.Tensor] = []

    # Query loop: simulate multiple query minibatches and accumulate objective.
    for _ in range(query_steps):
        query_embed = (
            torch.randn(state.q, state.d, device=state.device, dtype=torch.float32)
            .contiguous()
            .requires_grad_(True)
        )
        query_vert_idx = torch.randint(
            0, state.v, (state.q, 3), device=state.device, dtype=torch.int64
        )
        query_bary = _random_barycentric(state.q, state.device).requires_grad_(True)

        t_query_total = _event_start()

        t0 = _event_start()
        _, src_knn_idx = state.faiss_index.search(query_embed.detach(), k=state.k)
        ms_search = _event_elapsed_ms(t0)

        t0 = _event_start()
        knn_dist = state.faiss_index.knn_distances(query_embed, src_knn_idx)
        ms_dist = _event_elapsed_ms(t0)

        t0 = _event_start()
        weights_post = state.weighter.compute(knn_dist)
        weights_post.retain_grad()
        ms_weights = _event_elapsed_ms(t0)

        t0 = _event_start()
        evals_knn, evecs_knn, _ = state.gather.gather_queries(
            src_knn_idx=src_knn_idx,
            m_neigh_idx=m_neigh_idx,
            m_neigh_w=m_neigh_w,
            vert_idx=query_vert_idx,
            bary_coords=query_bary,
            evals=state.evals_grid,
            evecs=state.evecs_grid,
        )
        ms_gather = _event_elapsed_ms(t0)

        t0 = _event_start()
        heat_qk, heat_qk_norm = state.heat.query(
            src_knn_idx=src_knn_idx,
            evals_knn=evals_knn,
            evecs_knn=evecs_knn,
            weights_post_diff=weights_post,
            normalize=True,
        )
        ms_heat = _event_elapsed_ms(t0)

        assert heat_qk_norm is not None
        loss = loss + heat_qk.mean() + 0.1 * heat_qk_norm.square().mean()

        last_shapes = {
            "src_knn_idx": tuple(src_knn_idx.shape),
            "knn_dist": tuple(knn_dist.shape),
            "weights_post": tuple(weights_post.shape),
            "evals_knn": tuple(evals_knn.shape),
            "evecs_knn": tuple(evecs_knn.shape),
            "heat_qk": tuple(heat_qk.shape),
            "heat_qk_norm": tuple(heat_qk_norm.shape),
        }

        if record_timings:
            timings["query_search"].append(ms_search)
            timings["query_dist"].append(ms_dist)
            timings["query_weights"].append(ms_weights)
            timings["query_gather"].append(ms_gather)
            timings["query_heat"].append(ms_heat)
            timings["query_total"].append(_event_elapsed_ms(t_query_total))

        query_embeds.append(query_embed)
        query_barys.append(query_bary)
        weights_posts.append(weights_post)

    return loss, timings, last_shapes, query_embeds, query_barys, weights_posts


def main() -> None:
    args = _parse_args()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example (FAISS GPU path).")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")
    print("device:", device)
    print("matmul_precision:", torch.get_float32_matmul_precision())
    print("allow_tf32.matmul:", torch.backends.cuda.matmul.allow_tf32)
    print("allow_tf32.cudnn:", torch.backends.cudnn.allow_tf32)

    state = build_state(args, device)
    optimizer = torch.optim.Adam(
        [state.src_embed, state.src_angles, state.src_scales, state.src_bary],
        lr=1e-3,
    )

    # Warmup pass to trigger compilation/caches.
    optimizer.zero_grad(set_to_none=True)
    warm_steps = max(1, args.warmup_query_steps)
    warm_loss, _, _, _, _, _ = run_pipeline_iteration(
        state,
        query_steps=warm_steps,
        record_timings=False,
    )
    warm_loss.backward()
    optimizer.zero_grad(set_to_none=True)

    # Timed training-style pass: zero_grad -> accumulate over query loop -> backward -> step.
    optimizer.zero_grad(set_to_none=True)
    timed_steps = max(1, args.query_steps)
    loss, timings, shapes, query_embeds, query_barys, weights_posts = run_pipeline_iteration(
        state,
        query_steps=timed_steps,
        record_timings=True,
    )
    loss.backward()

    print("shapes (last query step):")
    for k, v in shapes.items():
        print(f"  {k}: {v}")

    print("timing stats:")
    _print_stats("faiss_build", timings["faiss_build"])
    _print_stats("select_grid", timings["select_grid"])
    _print_stats("gather_src", timings["gather_src"])
    _print_stats("heat_build", timings["heat_build"])
    _print_stats("query_search", timings["query_search"])
    _print_stats("query_dist", timings["query_dist"])
    _print_stats("query_weights", timings["query_weights"])
    _print_stats("query_gather", timings["query_gather"])
    _print_stats("query_heat", timings["query_heat"])
    _print_stats("query_total", timings["query_total"])

    print("grads (after accumulated backward across query loop):")
    print("  grad src_embed:", state.src_embed.grad is not None)
    print("  grad src_angles:", state.src_angles.grad is not None)
    print("  grad src_scales:", state.src_scales.grad is not None)
    print("  grad src_bary:", state.src_bary.grad is not None)
    print("  grad all query_embed:", all(t.grad is not None for t in query_embeds))
    print("  grad all query_bary:", all(t.grad is not None for t in query_barys))
    print("  grad all weights_post:", all(t.grad is not None for t in weights_posts))

    optimizer.step()


if __name__ == "__main__":
    main()
