import torch
from dataclasses import dataclass
from termcolor import colored

import heatsplats
from heatsplats.utils.typing import *
from heatsplats.density_controllers.base import BaseDensityController


@heatsplats.register("density_controllers.opacity")
class OpacityController(BaseDensityController):
    """Density controller that adjusts the number of kernels based on the opacity"""

    @dataclass
    class Config(BaseDensityController.Config):
        prune_opacity: float | None = 0.05
        prune_interval: int | None = 100
        reset_opacity_interval: int | None = 3000

    cfg: Config

    def check_sanity(self):
        super().check_sanity()
        if self.cfg.prune_opacity is not None:
            assert self.cfg.prune_opacity >= 0 and self.cfg.prune_opacity <= 1

        assert "_opacities" in self._trainable_params_names, (
            "The model must have a trainable parameter named '_opacities' to ",
            "use OpacityController",
        )

    def pre_backward_step(self, *args, **kwargs):
        print("Pre-backward hook called in OpacityController")

    def post_backward_step(self, step, *args, **kwargs):
        if self.cfg.stop_iter is not None and step >= self.cfg.stop_iter:
            return

        if step >= self.cfg.start_iter:
            if (
                self.cfg.prune_opacity is not None
                and self.cfg.prune_interval is not None
                and step % self.cfg.prune_interval == 0
            ):
                self.remove_transparent()

            if (
                self.cfg.reset_opacity_interval is not None
                and step % self.cfg.reset_opacity_interval == 0
            ):
                self.reset_opacities(value=self.cfg.prune_opacity * 2.0)

    @torch.no_grad()
    def remove_transparent(self):
        opacities = torch.abs(self._params["_opacities"].flatten())
        is_prune = opacities <= self.cfg.prune_opacity
        self._remove_kernels(is_prune)

        if heatsplats.is_debug():
            heatsplats.debug(
                colored(
                    f"Pruning {is_prune.sum().item()} / {len(opacities)} kernels "
                    f"with opacity <= {self.cfg.prune_opacity}",
                    "light_blue",
                )
            )

    @torch.no_grad()
    def reset_opacities(self, value: float):
        """Inplace reset the opacities to a given value"""

        def param_fn(name: str, p: Tensor) -> Tensor:
            if name == "_opacities":
                opacities = torch.clamp(
                    p,
                    min=torch.tensor(-value).item(),
                    max=torch.tensor(value).item(),
                )
                return torch.nn.Parameter(opacities, requires_grad=p.requires_grad)
            else:
                raise ValueError(f"Unexpected parameter name: {name}")

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            return torch.zeros_like(v)

        self._update_param_with_optimizer(param_fn, optimizer_fn, names=["_opacities"])

        if heatsplats.is_debug():
            heatsplats.debug(
                colored(f"Reset opacities to {-value} <= op <= {value}", "light_blue")
            )
