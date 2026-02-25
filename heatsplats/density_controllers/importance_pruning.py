import torch
from dataclasses import dataclass
from termcolor import colored

import heatsplats
from heatsplats.utils.typing import *
from heatsplats.density_controllers.base import BaseDensityController


@heatsplats.register("density_controllers.importance_pruning")
class ImportancePruningController(BaseDensityController):
    """
    Pruning strategy for normalized ImageGS-style splatting.
    Tracks 'hits' (top-k inclusions) and total weight contribution.
    """

    @dataclass
    class Config(BaseDensityController.Config):
        prune_interval: int = 100
        accumulation_interval: int = 100

        selection_threshold: float = 0.001
        contrib_threshold: float = 0.01
        min_kernels: int = 100

    cfg: Config

    def _post_configure(self, *args, **kwargs):
        num_kernels = self._model.N_sources
        device = self._model.device

        # Persistent hit counter for top-k inclusion
        hit_accumulator = torch.zeros(num_kernels, 1, device=device)
        self._model.register_buffer("_hit_accumulator", hit_accumulator)

        # Energy accumulator (sum of filtered weights)
        contribution_accumulator = torch.zeros(num_kernels, 1, device=device)
        self._model.register_buffer("_contrib_accumulator", contribution_accumulator)

    def check_sanity(self):
        super().check_sanity()
        assert self.cfg.prune_interval > 0
        assert self.cfg.accumulation_interval > 0
        assert self.cfg.selection_threshold >= 0 and self.cfg.selection_threshold <= 1.0
        assert self.cfg.contrib_threshold >= 0.0

    def pre_backward_step(
        self,
        step: int,
        kernel_contributions: Float[Tensor, "G P 1"],
        topk_kernel_idxs: Int[Tensor, "kG"],
        **kwargs,
    ):
        is_active = step >= (self.cfg.start_iter - self.cfg.accumulation_interval)
        if self.cfg.stop_iter is not None:
            is_active = is_active and step < self.cfg.stop_iter

        if not is_active:
            return

        with torch.no_grad():
            idx_flat = topk_kernel_idxs.reshape(-1, 1)
            ones = torch.ones_like(idx_flat, dtype=torch.float32)

            self._model._hit_accumulator.scatter_add_(dim=0, index=idx_flat, src=ones)

            topk_contribs = torch.gather(
                kernel_contributions, dim=0, index=topk_kernel_idxs
            )

            contribs_flat = topk_contribs.reshape(-1, 1)
            self._model._contrib_accumulator.scatter_add_(
                dim=0, index=idx_flat, src=contribs_flat
            )

    def post_backward_step(self, step, *args, **kwargs):
        if step < self.cfg.start_iter:
            return

        if step % self.cfg.prune_interval == 0:
            self.prune_redundant_kernels()
            self._model._hit_accumulator.zero_()
            self._model._contrib_accumulator.zero_()
            torch.cuda.empty_cache()  # Free up memory after pruning

    @torch.no_grad()
    def prune_redundant_kernels(self):
        steps_accumulated = self.cfg.accumulation_interval

        hits = self._model._hit_accumulator.flatten()
        avg_contribs = self._model._contrib_accumulator.flatten() / steps_accumulated

        # Prune if it covers a negligible percentage of the area
        # or if its total influence (contribution) is tiny.
        is_unused = hits < (self.cfg.selection_threshold * hits.max().item())
        is_contribless = avg_contribs < self.cfg.contrib_threshold

        prune_mask = is_unused | is_contribless

        # Ensure we maintain a minimum number of kernels
        num_kernels = self._model.N_sources
        num_to_prune = prune_mask.sum().item()

        if num_kernels - num_to_prune < self.cfg.min_kernels:
            num_allowed_to_prune = max(0, num_kernels - self.cfg.min_kernels)

            if num_allowed_to_prune == 0:
                prune_mask[:] = False
            else:
                # We need to prune fewer kernels than identified.
                # We prioritize pruning those with the lowest contribution.
                prune_indices = torch.where(prune_mask)[0]
                scores = avg_contribs[prune_indices]
                _, indices_to_prune_local = torch.topk(
                    scores, k=num_allowed_to_prune, largest=False
                )
                indices_to_prune = prune_indices[indices_to_prune_local]
                prune_mask = torch.zeros_like(prune_mask)
                prune_mask[indices_to_prune] = True

        num_before = len(prune_mask)
        self._remove_kernels(prune_mask)

        if heatsplats.is_debug():
            heatsplats.debug(
                colored(
                    f"Selection Pruning: {prune_mask.sum().item()} / {num_before} "
                    f"kernels removed. "
                    f"{is_unused.sum().item()} considered unused, "
                    f"{is_contribless.sum().item()} considered contribution-less.",
                    "light_blue",
                )
            )
