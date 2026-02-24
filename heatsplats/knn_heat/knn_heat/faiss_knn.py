from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

from .config import FaissGpuIndexConfig
from . import _ops

__all__ = ["FaissGpuFlatIndex"]


class FaissGpuFlatIndex:
    """
    Two-phase FAISS GPU flat index for inner-minibatch reuse.

    Typical training pattern:
      1) build(database) once per minibatch
      2) search(queries, k) many times inside the minibatch loop
      3) recompute distances in torch for STE gradients via knn_distances()

    Notes:
    - search() uses FAISS and is not differentiable w.r.t. neighbor selection
    - knn_distances() recomputes distances for the selected neighbors using torch ops,
      enabling straight-through gradients through distance values (and into queries/database)
    - metric="l2": distances are squared L2 (matches FAISS)
    - metric="ip": distances are inner products (larger is closer)
    """

    def __init__(self, config: FaissGpuIndexConfig | None = None) -> None:
        self.config = config or FaissGpuIndexConfig()

        try:
            import faiss  # type: ignore
            import faiss.contrib.torch_utils  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "FAISS with GPU support is required. Install faiss-gpu and ensure CUDA is available. "
                "Also ensure faiss.contrib.torch_utils is importable."
            ) from exc

        import faiss  # type: ignore

        self._faiss = faiss
        self._res = faiss.StandardGpuResources()
        self._index: Optional[object] = None
        self._d: Optional[int] = None
        self._n: Optional[int] = None
        self._db: Optional[torch.Tensor] = None
        self._db_norm: Optional[torch.Tensor] = None

        self._dist_fn = self._make_distance_fn()
        self._dist_fn_with_db_norm = self._make_distance_fn_with_db_norm()

    def _make_distance_fn(self):
        if self.config.distance_impl == "naive":
            if self.config.metric == "l2":
                fn = _ops._knn_distances_l2_sq
            elif self.config.metric == "ip":
                fn = _ops._knn_distances_ip
            else:
                raise ValueError(f"Unsupported metric: {self.config.metric}")
        elif self.config.distance_impl == "bmm":
            if self.config.metric == "l2":
                fn = _ops._knn_distances_l2_sq_bmm
            elif self.config.metric == "ip":
                fn = _ops._knn_distances_ip_bmm
            else:
                raise ValueError(f"Unsupported metric: {self.config.metric}")
        else:
            raise ValueError(f"Unsupported distance_impl: {self.config.distance_impl}")

        if self.config.compile_distances and hasattr(torch, "compile"):
            return torch.compile(fn, dynamic=True)

        return fn

    def _make_distance_fn_with_db_norm(self):
        if self.config.metric != "l2":
            return None

        fn = _ops._knn_distances_l2_sq_bmm_with_db_norm

        if self.config.compile_distances and hasattr(torch, "compile"):
            return torch.compile(fn, dynamic=True)

        return fn

    @staticmethod
    def _check_cuda_2d(x: torch.Tensor, name: str) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not x.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if x.ndim != 2:
            raise ValueError(f"{name} must be 2D [N, D], got shape {tuple(x.shape)}")
        if not x.is_contiguous():
            raise ValueError(f"{name} must be contiguous (call contiguous())")
        if x.dtype not in (torch.float16, torch.float32):
            raise ValueError(f"{name} must be float16 or float32, got {x.dtype}")

    @staticmethod
    def _check_indices(indices: torch.Tensor) -> None:
        if not isinstance(indices, torch.Tensor):
            raise TypeError("indices must be a torch.Tensor")
        if indices.ndim != 2:
            raise ValueError(
                f"indices must be 2D [M, K], got shape {tuple(indices.shape)}"
            )
        if not indices.is_cuda:
            raise ValueError("indices must be a CUDA tensor")
        if indices.dtype not in (torch.int64, torch.int32):
            raise ValueError(f"indices must be int64 or int32, got {indices.dtype}")

    def reset(self) -> None:
        self._index = None
        self._d = None
        self._n = None
        self._db = None
        self._db_norm = None

    def build(self, database: torch.Tensor) -> None:
        """
        Build (or rebuild) the FAISS GPU index from a database tensor.

        This method uploads the database vectors to a flat FAISS index on the GPU.
        It is intended to be called once per outer or inner minibatch, after which
        search() can be invoked multiple times with different query tensors.

        For metric="l2", an optional cache of per-database squared norms (||x||^2)
        can be constructed when build_db_norm=True. This cache is used only by
        knn_distances() when distance_impl="bmm" to reduce redundant computation
        during straight-through distance recomputation. It does not affect FAISS
        search results.

        Args:
            database: [N, D] float16/float32 CUDA tensor containing the database
                embeddings to index

        Returns:
            None
        """
        self._check_cuda_2d(database, "database")
        d = int(database.shape[1])
        n = int(database.shape[0])

        if self.config.metric == "l2":
            metric = self._faiss.METRIC_L2
            quantizer = self._faiss.IndexFlatL2(d)
        elif self.config.metric == "ip":
            metric = self._faiss.METRIC_INNER_PRODUCT
            quantizer = self._faiss.IndexFlatIP(d)
        else:
            raise ValueError(f"Unsupported metric: {self.config.metric}")

        if self.config.index_type == "flat":
            gpu_cfg = self._faiss.GpuIndexFlatConfig()
            gpu_cfg.useFloat16 = bool(self.config.use_float16)
            gpu_index = self._faiss.GpuIndexFlat(self._res, d, metric, gpu_cfg)
            gpu_index.add(database)
        elif self.config.index_type == "ivf_flat":
            if n < self.config.ivf_nlist:
                raise ValueError("database size must be >= ivf_nlist")

            cpu_ivf = self._faiss.IndexIVFFlat(
                quantizer, d, int(self.config.ivf_nlist), metric
            )
            gpu_index = self._faiss.index_cpu_to_gpu(self._res, 0, cpu_ivf)
            gpu_index.train(database)  # required for IVF
            gpu_index.nprobe = int(self.config.ivf_nprobe)
            gpu_index.add(database)
        elif self.config.index_type == "cagra":
            if self.config.metric != "l2":
                raise ValueError("CAGRA is typically L2-only in this setup")

            cagra_cfg = self._faiss.GpuIndexCagraConfig()
            cagra_cfg.use_cuvs = True
            cagra_cfg.graph_degree = int(self.config.cagra_graph_degree)
            cagra_cfg.intermediate_graph_degree = int(
                self.config.cagra_intermediate_graph_degree
            )
            cagra_cfg.nn_descent_niter = int(self.config.cagra_nn_descent_niter)
            cagra_cfg.refine_rate = float(self.config.cagra_refine_rate)
            cagra_cfg.build_algo = {
                "nn_descent": self._faiss.graph_build_algo_NN_DESCENT,
                "iterative_search": self._faiss.graph_build_algo_ITERATIVE_SEARCH,
                "ivf_pq": self._faiss.graph_build_algo_IVF_PQ,
            }[self.config.cagra_build_algo]

            gpu_index = self._faiss.GpuIndexCagra(self._res, d, metric, cagra_cfg)

            # CAGRA builds graph at train-time for full dataset
            gpu_index.train(database)

        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")

        self._index = gpu_index
        self._d = d
        self._n = n
        self._db = database

        # Optional cache for L2: ||x||^2 for each database vector. This is only used by the
        # distance_impl="bmm" path, and only affects knn_distances() (not FAISS search()).
        if self.config.build_db_norm and self.config.metric == "l2":
            self._db_norm = (self._db * self._db).sum(dim=-1)
        else:
            self._db_norm = None

    def search(
        self, queries: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search the already-built index.

        Args:
            queries: [M, D] float16/float32 CUDA tensor
            k: number of neighbors

        Returns:
            distances: [M, K] CUDA tensor (float32)
            indices:   [M, K] CUDA tensor (typically int64)
        """
        if self._index is None or self._d is None or self._n is None:
            raise RuntimeError("Index is not built. Call build(database) first.")

        self._check_cuda_2d(queries, "queries")

        if int(queries.shape[1]) != self._d:
            raise ValueError(
                f"Dim mismatch: queries D={queries.shape[1]} vs index D={self._d}"
            )
        if k <= 0:
            raise ValueError("k must be positive")
        if k > self._n:
            raise ValueError("k cannot exceed database size")

        distances, indices = self._index.search(queries, int(k))
        return distances, indices

    def knn_distances(
        self,
        queries: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable recomputation of distances using torch ops.

        This enables straight-through gradients through the *distance values*
        after using FAISS indices from search().

        Args:
            queries:  [M, D] float16/float32 CUDA tensor (grad ok)
            indices:  [M, K] int32/int64 CUDA tensor from FAISS

        Returns:
            dists: [M, K] float tensor on CUDA
        """
        if self._db is None:
            raise RuntimeError("Database is not built. Call build(database) first.")

        self._check_cuda_2d(queries, "queries")
        self._check_cuda_2d(self._db, "database")
        self._check_indices(indices)

        if queries.shape[1] != self._db.shape[1]:
            raise ValueError(
                f"Dim mismatch: queries D={queries.shape[1]} vs database D={self._db.shape[1]}"
            )
        if self._d is not None and int(queries.shape[1]) != self._d:
            raise ValueError(
                f"Dim mismatch: queries D={queries.shape[1]} vs index D={self._d}"
            )

        # Ensure indexing dtype is long for torch advanced indexing
        if indices.dtype != torch.int64:
            indices = indices.to(torch.int64)

        if (
            self.config.build_db_norm
            and self.config.metric == "l2"
            and self.config.distance_impl == "bmm"
        ):
            if self._db_norm is None:
                raise RuntimeError(
                    "db_norm is not built. Set build_db_norm=True and call build(database) first."
                )
            return self._dist_fn_with_db_norm(queries, self._db, indices, self._db_norm)

        return self._dist_fn(queries, self._db, indices)
