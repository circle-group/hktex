from __future__ import annotations

import argparse

import torch

from .heat_diff_old import heat_diffusion
from knn_heat import SpectralKnnHeat, SpectralKnnHeatConfig


def _parse_bool(x: str) -> bool:
    v = x.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare old all-pairs heat diffusion vs SpectralKnnHeat outputs/gradients."
    )
    parser.add_argument("--s", type=int, default=32, help="Number of sources.")
    parser.add_argument(
        "--q",
        type=int,
        default=32,
        help="Number of queries. For strict all-sources-as-queries, set q=s.",
    )
    parser.add_argument("--e", type=int, default=48, help="Spectral dimension.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Normalization epsilon.")
    parser.add_argument(
        "--use-post-weights",
        type=_parse_bool,
        default=True,
        help="Apply weights_post_diff in both old and new paths (true/false).",
    )
    parser.add_argument(
        "--compile-heat",
        type=_parse_bool,
        default=True,
        help="Whether to torch.compile heat ops (true/false).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
        help="Comparison dtype.",
    )
    return parser.parse_args()


def _max_abs(x: torch.Tensor) -> float:
    return float(x.abs().max().item())


def main() -> None:
    args = _parse_args()
    torch.manual_seed(0)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("dtype:", dtype)

    s = args.s
    q = args.q
    e = args.e

    # Shared initial tensors.
    init_evals_src = torch.rand(s, e, device=device, dtype=dtype)
    init_evals_query = torch.rand(s, q, e, device=device, dtype=dtype)
    init_evecs_src = torch.randn(s, e, device=device, dtype=dtype)
    init_evecs_query = torch.randn(s, q, e, device=device, dtype=dtype)
    init_weights_query = torch.rand(s, q, device=device, dtype=dtype)
    time = torch.full((s,), 0.37, device=device, dtype=dtype)

    # ----- Old path -----
    evals_src_old = init_evals_src.clone().requires_grad_(True)
    evals_query_old = init_evals_query.clone().requires_grad_(True)
    evecs_src_old = init_evecs_src.clone().requires_grad_(True)
    evecs_query_old = init_evecs_query.clone().requires_grad_(True)
    weights_query_old = init_weights_query.clone().requires_grad_(True)

    # Make effective evals depend on both source and query eval parameters
    # so we can compare gradients for both.
    evals_eff_old = evals_src_old + evals_query_old.mean(dim=1)  # [S, E]

    # Build old evec tensor [S, Q+1, E]: first Q are query points, last is source.
    evecs_old = torch.cat(
        [evecs_query_old, evecs_src_old.unsqueeze(1)],
        dim=1,
    )  # [S, Q+1, E]

    # Dirac at the source slot (last index) for each source batch.
    x_old = torch.zeros(s, q + 1, 1, device=device, dtype=dtype)
    x_old[:, q, 0] = 1.0
    mass_old = torch.ones(s, q + 1, device=device, dtype=dtype)
    weights_post_old = None
    if args.use_post_weights:
        # Old contract: source/self slot is 1 for normalization reference.
        weights_post_old = torch.cat(
            [
                weights_query_old,
                torch.ones(s, 1, device=device, dtype=dtype),
            ],
            dim=1,
        )  # [S, Q+1]

    diff_old = heat_diffusion(
        x=x_old,
        mass=mass_old,
        evals=evals_eff_old,
        evecs=evecs_old,
        time=time,
        weights_post_diff=weights_post_old,
    )
    num_old_sq = diff_old[:, :q, 0]
    den_old_s = diff_old[:, q, 0]
    norm_old_sq = num_old_sq / (den_old_s.unsqueeze(1) + args.eps)
    loss_old = num_old_sq.mean() + 0.1 * norm_old_sq.square().mean()
    loss_old.backward()

    # ----- New path (all-pairs via K=S) -----
    evals_src_new = init_evals_src.clone().requires_grad_(True)
    evals_query_new = init_evals_query.clone().requires_grad_(True)
    evecs_src_new = init_evecs_src.clone().requires_grad_(True)
    evecs_query_new = init_evecs_query.clone().requires_grad_(True)
    weights_query_new = init_weights_query.clone().requires_grad_(True)
    evals_eff_new = evals_src_new + evals_query_new.mean(dim=1)  # [S, E]

    # KNN layout for all-pairs: each query sees all sources in fixed order.
    src_knn_idx = (
        torch.arange(s, device=device, dtype=torch.int64).view(1, s).expand(q, s)
    )

    # Map old [S,Q,E] tensors to new [Q,S,E] with K=S.
    evals_knn_qse = evals_eff_new.unsqueeze(0).expand(q, s, e)
    evecs_knn_qse = evecs_query_new.permute(1, 0, 2).contiguous()
    evecs_src_knn_qse = evecs_src_new.unsqueeze(0).expand(q, s, e)
    weights_post_qs = None
    if args.use_post_weights:
        # Old query weights are [S,Q]; new expects [Q,S].
        weights_post_qs = weights_query_new.transpose(0, 1).contiguous()

    heat = SpectralKnnHeat(
        SpectralKnnHeatConfig(
            compile_heat=args.compile_heat,
            eps=args.eps,
        )
    )
    heat.build(evals_src=evals_eff_new, evecs_src=evecs_src_new, time=time)
    num_new_qs, norm_new_qs = heat.query(
        src_knn_idx=src_knn_idx,
        evals_knn=evals_knn_qse,
        evecs_knn=evecs_knn_qse,
        evecs_src_knn=evecs_src_knn_qse,
        weights_post_diff=weights_post_qs,
        normalize=True,
    )

    # Back to [S,Q] for direct comparison with old path.
    num_new_sq = num_new_qs.transpose(0, 1).contiguous()
    assert norm_new_qs is not None
    norm_new_sq = norm_new_qs.transpose(0, 1).contiguous()

    loss_new = num_new_sq.mean() + 0.1 * norm_new_sq.square().mean()
    loss_new.backward()

    # ----- Output comparison -----
    print("max_abs(num_old - num_new):", _max_abs(num_old_sq - num_new_sq))
    print("max_abs(norm_old - norm_new):", _max_abs(norm_old_sq - norm_new_sq))
    print("abs(loss_old - loss_new):", float((loss_old - loss_new).abs().item()))

    # ----- Gradient comparison -----
    print(
        "max_abs(grad evals_src):",
        _max_abs(evals_src_old.grad - evals_src_new.grad),
    )
    print(
        "max_abs(grad evecs_src):",
        _max_abs(evecs_src_old.grad - evecs_src_new.grad),
    )
    print(
        "max_abs(grad evals_query):",
        _max_abs(evals_query_old.grad - evals_query_new.grad),
    )
    print(
        "max_abs(grad evecs_query):",
        _max_abs(evecs_query_old.grad - evecs_query_new.grad),
    )
    if args.use_post_weights:
        print(
            "max_abs(grad weights_post_diff_query):",
            _max_abs(weights_query_old.grad - weights_query_new.grad),
        )

    # Quick finite checks for safety.
    finite_ok = bool(
        torch.isfinite(num_new_qs).all().item()
        and torch.isfinite(norm_new_qs).all().item()
        and torch.isfinite(evals_src_new.grad).all().item()
        and torch.isfinite(evecs_src_new.grad).all().item()
        and torch.isfinite(evals_query_new.grad).all().item()
        and torch.isfinite(evecs_query_new.grad).all().item()
    )
    print("finite_ok:", finite_ok)


if __name__ == "__main__":
    main()
