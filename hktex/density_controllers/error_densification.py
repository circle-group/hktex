import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from termcolor import colored

import hktex
from hktex.utils.typing import *
from hktex.modules import (
    EigenAlboInterpolation,
    GeodesicTracer,
    KernelInfo,
)
from .base import BaseDensityController


@hktex.register("density_controllers.error_based_densification")
class ErrorDensificationController(BaseDensityController):
    """
    This controller introduces a learnable scalar 'e_k' for each kernel. It uses an
    auxiliary loss to accumulate the total rendering error associated with each
    kernel into the gradient of 'e_k'. This per-kernel error is then used to
    decide which kernels to split (if they are large) or clone (if they are small).
    """

    @dataclass
    class Config(BaseDensityController.Config):
        densify_interval: int = 100
        error_accumulation_interval: int = 100
        error_threshold: float = 0.0002

        size_threshold: float = 0.1  # Distinguish between splitting and cloning
        split_radius: float = 0.05  # When splitting, how far to displace new kernels
        max_densify_ratio: float = 0.05  # Max 5% new kernels at each densification
        max_kernels: int = 3_000

    cfg: Config

    def _post_configure(self, *args, **kwargs):
        num_kernels = self._model.N_sources
        device = self._model.device
        accumulator = torch.zeros(num_kernels, 1, device=device)
        self._model.register_buffer("_error_accumulator", accumulator)

    def check_sanity(self):
        super().check_sanity()
        assert self.cfg.densify_interval > 0
        assert self.cfg.error_accumulation_interval > 0
        assert self.cfg.size_threshold >= 0 and self.cfg.size_threshold <= 1.0
        assert self.cfg.error_threshold >= 0.0
        assert self.cfg.max_densify_ratio > 0.0 and self.cfg.max_densify_ratio < 1.0
        assert self.cfg.max_kernels >= self._model.N_sources

    def pre_backward_step(
        self,
        step: int,
        rendered_colours: Float[Tensor, "P D"],
        gt_colours: Float[Tensor, "P D"],
        kernel_contributions: Optional[Float[Tensor, "G P 1"]],
        topk_kernel_idxs: Int[Tensor, "k P 1"],
        topk_kernel_contribs: Optional[Float[Tensor, "k P 1"]] = None,
        *args,
        **kwargs,
    ):
        """
        Computes and backpropagates the auxiliary loss to populate _errors.grad.
        While the densification happens every `densify_interval` steps, this
        auxiliary loss is computed at every step to keep accumulating the error.

        Args:
            rendered_colours: The output of the model [P, D].
            gt_colours: The ground truth colours [P, D].
            eigalbo_interp: The EigenAlboInterpolation instance.
            points_info: The PointsInfo for the current batch. Computed in trainer.
            kernel_info: The KernelInfo for the current model. Computed in trainer.
        """
        is_active = step >= (self.cfg.start_iter - self.cfg.error_accumulation_interval)
        if self.cfg.stop_iter is not None:
            is_active = is_active and step < self.cfg.stop_iter

        if not is_active:
            return

        with torch.no_grad():
            per_point_error = torch.abs(rendered_colours - gt_colours).mean(dim=1)

            if topk_kernel_contribs is not None:
                topk_contribs = topk_kernel_contribs
            else:
                if kernel_contributions is None:
                    raise ValueError(
                        "Need either dense kernel contributions or topk kernel contributions"
                    )
                topk_contribs = torch.gather(
                    kernel_contributions, dim=0, index=topk_kernel_idxs
                )  # [k, P, 1]

            # Normalize the weights to distribute errors based on actual influence
            normalization = topk_contribs.sum(dim=0, keepdim=True) + 1e-8  # [1,P,1]
            normalized_contribs = topk_contribs / normalization  # [k,P,1]

            # fmt: off
            weighted_error = normalized_contribs.squeeze(-1) \
                * per_point_error.unsqueeze(0)  # [k,P]
            # fmt: on
            self._model._error_accumulator.scatter_add_(
                dim=0,
                index=topk_kernel_idxs.reshape(-1, 1),
                src=weighted_error.reshape(-1, 1),
            )

    def post_backward_step(
        self,
        step,
        eigalbo_interp: EigenAlboInterpolation,
        kernel_info: KernelInfo,
        tracer: GeodesicTracer,
        *args,
        **kwargs,
    ):
        """
        After the main backward pass, check if it's time to densify and then
        reset the error accumulation.
        """
        is_densify_step = (
            step >= self.cfg.start_iter and step % self.cfg.densify_interval == 0
        )
        if self.cfg.stop_iter is not None:
            is_densify_step = is_densify_step and step < self.cfg.stop_iter

        if not is_densify_step:
            return

        self._model._error_accumulator /= self.cfg.error_accumulation_interval

        self._densify(eigalbo_interp, kernel_info, tracer)

        self._model._error_accumulator.zero_()
        torch.cuda.empty_cache()  # Free up memory after densification

    @torch.no_grad()
    def _densify(
        self,
        eigalbo_interp: EigenAlboInterpolation,
        kernel_info: KernelInfo,
        tracer: GeodesicTracer,
    ):
        # per_kernel_error = self._model._errors.grad
        per_kernel_error = self._model._error_accumulator
        if per_kernel_error is None:
            return
        per_kernel_error = per_kernel_error.abs().flatten()

        # Identify candidates for densification
        candidate_mask = per_kernel_error > self.cfg.error_threshold
        num_candidates = candidate_mask.sum().item()

        if num_candidates == 0:
            if hktex.is_debug():
                hktex.debug(
                    colored(
                        f"No densification. There were no candidates with error > "
                        f"{self.cfg.error_threshold}. The mean error was "
                        f"{per_kernel_error.mean().item()}.",
                        "light_green",
                    )
                )
            return

        # Apply caps on how many kernels can be added
        max_new_from_ratio = int(self._model.N_sources * self.cfg.max_densify_ratio)
        room_for_new = self.cfg.max_kernels - self._model.N_sources
        num_to_add = min(num_candidates, max_new_from_ratio, room_for_new)

        if num_to_add <= 0:
            return

        # Prioritize candidates with the highest error
        candidate_indices = torch.where(candidate_mask)[0]
        if num_candidates > num_to_add:
            candidate_errors = per_kernel_error[candidate_indices]
            _, top_indices = torch.topk(candidate_errors, num_to_add)
            final_indices = candidate_indices[top_indices]
        else:
            final_indices = candidate_indices

        densify_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        densify_mask[final_indices] = True

        # Distinguish between splitting and cloning based on size
        # Using the more intuitive inverse relationship: 1.0 - epsilon
        kernel_radii = 1.0 - self._model.thresholds.flatten().detach()
        is_large = kernel_radii > self.cfg.size_threshold

        split_mask = densify_mask & is_large
        clone_mask = densify_mask & ~is_large

        if clone_mask.any():
            self.clone(clone_mask)

            # Pad the original split_mask with `False` for the newly added kernels
            num_cloned = clone_mask.sum().item()
            padding = torch.zeros(
                num_cloned, dtype=torch.bool, device=split_mask.device
            )
            split_mask = torch.cat([split_mask, padding])

        if split_mask.any():
            self.split(
                split_mask, eigalbo_interp, kernel_info, tracer, self.cfg.split_radius
            )

        if hktex.is_debug():
            hktex.debug(
                colored(
                    f"{self._model.N_sources} kernels: "
                    f"Cloned {clone_mask.sum()} and Split {split_mask.sum()} kernels. "
                    f"There were {per_kernel_error.numel()} candidates with error > "
                    f"{self.cfg.error_threshold}. The mean error was "
                    f"{per_kernel_error.mean().item()}.",
                    "light_green",
                )
            )
