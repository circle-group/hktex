from __future__ import annotations

import time

import torch

from knn_heat.faiss_knn import FaissGpuFlatIndex, FaissGpuIndexConfig


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example")

    device = torch.device("cuda")
    torch.manual_seed(0)

    d = 128
    k = 32
    n_db = 200_000
    n_q = 2_048

    # Keep the built database tensor differentiable so STE distances can backprop into it.
    database = (
        torch.randn(n_db, d, device=device, dtype=torch.float32)
        .contiguous()
        .requires_grad_(True)
    )
    queries = torch.randn(n_q, d, device=device, dtype=torch.float32).contiguous()

    # If you want cosine similarity:
    # database = l2_normalize(database).contiguous()
    # queries = l2_normalize(queries).contiguous()
    # cfg = FaissGpuIndexConfig(metric="ip", use_float16=False, compile_distances=True)

    cfg = FaissGpuIndexConfig(metric="l2", use_float16=False, compile_distances=True)
    index = FaissGpuFlatIndex(cfg)

    # Build once per minibatch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    index.build(database)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"build: {(t1 - t0) * 1e3:.2f} ms")

    # Example inner loop: many queries with same built index
    n_repeats = 5
    for step in range(n_repeats):
        distances_faiss, indices = index.search(queries, k=k)

        # Straight-through: recompute distances with torch so gradients flow
        queries_req = queries.detach().clone().requires_grad_(True)
        database.grad = None

        distances_ste = index.knn_distances(
            queries=queries_req,
            indices=indices,
        )

        # Dummy loss to prove gradients exist
        loss = distances_ste.mean()
        loss.backward()

        print(
                f"step {step}: "
                f"faiss_dists[{distances_faiss.dtype}] "
                f"ste_dists[{distances_ste.dtype}] "
                f"grad_q={queries_req.grad is not None} "
                f"grad_db={database.grad is not None}"
            )

    # Timing for search vs STE distance recomputation (single pass)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    distances_faiss, indices = index.search(queries, k=k)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    queries_req = queries.detach().clone().requires_grad_(True)
    database.grad = None

    torch.cuda.synchronize()
    t4 = time.perf_counter()
    distances_ste = index.knn_distances(queries_req, indices)
    torch.cuda.synchronize()
    t5 = time.perf_counter()
    loss = distances_ste.mean()
    loss.backward()

    print(f"search: {(t3 - t2) * 1e3:.2f} ms")
    print(f"ste distances: {(t5 - t4) * 1e3:.2f} ms")
    print(f"timing-pass grads: grad_q={queries_req.grad is not None} grad_db={database.grad is not None}")
    print("first query indices:", indices[0, :8].tolist())
    print("first query ste dists:", distances_ste[0, :8].tolist())


if __name__ == "__main__":
    main()
