import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer, ParamsT, required

from .tracer import GeodesicTracer
from hktex.utils.typing import *

__all__ = ["GeodesicOpt"]


class GeodesicOpt(optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        tracer: GeodesicTracer,
        lr: Union[float, Tensor] = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening, face_ids=required
        )
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
            momentum = group["momentum"]
            has_momentum = momentum != 0
            dampening = group["dampening"]

            # TODO: this check should be somewhere else
            assert len(group["params"]) == len(group["face_ids"])
            for p, face_id in zip(group["params"], group["face_ids"]):
                p: Tensor
                face_id: Tensor
                if p.grad is None:
                    continue
                grad = p.grad
                state: dict[str, Tensor] = self.state[p]

                momentum_buffer = None
                if has_momentum:
                    momentum_buffer = state.get("momentum_buffer")
                    if momentum_buffer is None:
                        momentum_buffer = grad.detach().clone()
                    else:
                        momentum_buffer.mul_(momentum).add_(grad, alpha=1 - dampening)
                    grad = momentum_buffer

                bary_coords = getattr(p, "bary_coords", None)
                self.tracer.trace_(
                    p,
                    face_id,
                    grad.mul(-lr),
                    bary_coords=bary_coords,
                    transport_vector=momentum_buffer,
                )

                if has_momentum:
                    state["momentum_buffer"] = momentum_buffer

        return loss
