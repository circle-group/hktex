import heatsplats
from .base import BaseDensityController
from heatsplats.utils.typing import *

__all__ = ["parse_density_controllers"]


def parse_density_controllers(config, model, optimizers) -> list[BaseDensityController]:
    density_controllers = []
    for cfg in config:
        dc = heatsplats.find(cfg.density_controller_type)(
            cfg.args, model=model, optimizers=optimizers
        )
        density_controllers.append(dc)
    return density_controllers
