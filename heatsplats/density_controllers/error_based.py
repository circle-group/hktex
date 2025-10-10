import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from termcolor import colored

import heatsplats
from heatsplats.utils.typing import *
from heatsplats.modules import (
    EigenAlboInterpolation,
    GeodesicTracer,
    KernelInfo,
)
from .base import BaseDensityController


@heatsplats.register("density_controllers.error_based_densification")
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
        max_densify_ratio: float = 0.05  # Max 5% new kernels at each densification
        max_kernels: int = 3_000

    cfg: Config

    def _post_configure(self, *args, **kwargs):
        # Add the auxiliary error parameter 'e_k' to the model
        num_kernels = self._model.N_sources
        device = self._model.device

        # errors_param = torch.nn.Parameter(
        #     torch.zeros(num_kernels, 1, device=device, requires_grad=True)
        # )
        # self._model.register_parameter("_errors", errors_param)
        # self._params["_errors"] = errors_param

        # Create a dedicated optimizer for the error parameters with zero LR. Not used
        # for optimization, just to to state management during densification strategies.
        # error_optimizer = UselessAdam(
        #     [{"params": [errors_param], "name": "_errors"}], lr=0.0
        # )
        # self._optimizers.append(error_optimizer)
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

        # assert "_errors" in self._trainable_params_names, (
        #     "The model must have a trainable parameter named '_errors' to ",
        #     "use ErrorDensificationController",
        # )

    def pre_backward_step(
        self,
        step: int,
        rendered_colours: Float[Tensor, "P D"],
        gt_colours: Float[Tensor, "P D"],
        kernel_contributions: Float[Tensor, "G P 1"],
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

        # Autograd method ##############################################################
        # # Detach previous grad to prevent creation of massive computational graph
        # if self._model._errors.grad is not None:
        #     self._model._errors.grad.detach_()

        # with torch.no_grad():
        #     per_point_error = torch.abs(rendered_colours - gt_colours).mean(dim=1)

        # contributions_reshaped = kernel_contributions.squeeze(-1).permute(1, 0)
        # rendered_errors = (
        #     contributions_reshaped @ self._model._errors
        # )  # [P, G] @ [G, 1] -> [P, 1]

        # aux_loss = torch.sum(per_point_error * rendered_errors)

        # # Populate self._model._errors.grad with the per-kernel error
        # aux_loss.backward(retain_graph=True)

        # Manual gradient computation to avoid creating a massive graph ################
        # TODO: Massive difference vs autograd. This has more sense, but why so different?
        with torch.no_grad():
            per_point_error = torch.abs(rendered_colours - gt_colours).mean(dim=1)
            kernel_contributions = kernel_contributions.squeeze(-1)
            kernel_error = (kernel_contributions @ per_point_error).unsqueeze(-1)
            self._model._error_accumulator += kernel_error

        # # Add the new gradient to the accumulator (.grad attribute)
        # if self._model._errors.grad is None:
        #     self._model._errors.grad = kernel_error
        # else:
        #     self._model._errors.grad += kernel_error

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

        # Reset the accumulated error gradient after each step or after densification
        # if self._model._errors.grad is not None:
        #     self._model._errors.grad.zero_()
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
            if heatsplats.is_debug():
                heatsplats.debug(
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
            self.split(split_mask, eigalbo_interp, kernel_info, tracer)

        if heatsplats.is_debug():
            heatsplats.debug(
                colored(
                    f"{self._model.N_sources} kernels: "
                    f"Cloned {clone_mask.sum()} and Split {split_mask.sum()} kernels. "
                    f"There were {per_kernel_error.numel()} candidates with error > "
                    f"{self.cfg.error_threshold}. The mean error was "
                    f"{per_kernel_error.mean().item()}.",
                    "light_green",
                )
            )
