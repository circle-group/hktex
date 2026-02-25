import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

import heatsplats
from heatsplats.modules import GeodesicOpt


def getattr_recursive(m, attr):
    for name in attr.split("."):
        m = getattr(m, name)
    return m


def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []


def parse_optimizer_and_scheduler(
    config, model
) -> tuple[torch.optim.Optimizer, _LRScheduler | None]:
    if hasattr(config, "params"):
        params = [
            {"params": get_parameters(model, name), "name": name, **args}
            for name, args in config.params.items()
        ]
        heatsplats.debug(f"Specify optimizer params: {config.params}")
    else:
        params = model.parameters()

    if config.name == "GeodesicOpt":
        for p in params:
            p["face_ids"] = [getattr_recursive(model, p["face_ids"])]
        tracer = getattr_recursive(model, config.args.get("tracer", "tracer"))
        optim = GeodesicOpt(
            params,
            tracer,
            lr=config.args.get("lr", 1e-3),
            momentum=config.args.get("momentum", 0),
            dampening=config.args.get("dampening", 0),
        )
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)

    scheduler = None
    if hasattr(config, "scheduler"):
        scheduler_config = config.scheduler
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_config.name)(
            optim, **scheduler_config.args
        )

    return optim, scheduler


def parse_optimizers_and_schedulers(config, model) -> list[torch.optim.Optimizer]:
    optims = []
    schedulers = []
    for optimizer in config:
        optim, scheduler = parse_optimizer_and_scheduler(optimizer, model)
        optims.append(optim)
        if scheduler is not None:
            schedulers.append(scheduler)
    return optims, schedulers
