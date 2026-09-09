import hktex
from .base import BaseDensityController
from hktex.utils.typing import *

__all__ = ["parse_density_controllers"]


def parse_density_controllers(
    config, mesh, model, optimizers
) -> list[BaseDensityController]:
    density_controllers = []
    for cfg in config:
        dc = hktex.find(cfg.density_controller_type)(
            cfg.args, mesh=mesh, model=model, optimizers=optimizers
        )
        density_controllers.append(dc)
    return density_controllers
