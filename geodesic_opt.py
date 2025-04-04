import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer, ParamsT, required

from .tracer import GeodesicTracer
from utils.typing import *


class GeodesicOpt(optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        tracer: GeodesicTracer,
        lr: Union[float, Tensor] = 1e-3,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, face_ids=required)
        super().__init__(params, defaults)

        self.tracer = tracer

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            # TODO: this check should be somewhere else
            assert len(group["params"]) == len(group["face_ids"])
            for p, face_id in zip(group["params"], group["face_ids"]):
                p: Tensor
                face_id: Tensor
                if p.grad is None:
                    continue
                d_p = p.grad

                p_n, face_id_n = self.tracer.trace(p, face_id, d_p.mul(-lr))
                p.copy_(p_n)
                face_id.copy_(face_id_n)

        return loss
