import torch
import torch.nn as nn

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


def parse_optimizer(config, model) -> torch.optim.Optimizer:
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
        optim = GeodesicOpt(params, tracer, lr=config.args.get("lr", 1e-3))
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim


def parse_optimizers(config, model) -> list[torch.optim.Optimizer]:
    optims = []
    for optimizer in config:
        optims.append(parse_optimizer(optimizer, model))
    return optims
