import torch

from .typing import *

__all__ = ["uniform_sample_triangle"]


def uniform_sample_triangle(uniform_samples: Float[Tensor, "B 2"]):
    su0 = uniform_samples[:, 0].sqrt()
    b0 = 1 - su0
    b1 = uniform_samples[:, 1] * su0
    return torch.stack((b0, b1, 1 - b0 - b1), dim=1)
