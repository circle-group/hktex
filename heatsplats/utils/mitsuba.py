from contextlib import contextmanager

import drjit as dr
import mitsuba as mi

from .typing import *

__all__ = ["dr_no_jit", "TraversableDict"]


# Trick from:  https://github.com/gerwang/diff-sdf/blob/9122375e37ca8757f77cd3fdc124e36b46c523be/python/util.py#L228
@contextmanager
def dr_no_jit(when=True):
    if when:
        old_loop_flag = dr.flag(dr.JitFlag.SymbolicLoops)
        old_call_flag = dr.flag(dr.JitFlag.SymbolicCalls)
        old_state_flag = dr.flag(dr.JitFlag.OptimizeCalls)
        old_conditional_flag = dr.flag(dr.JitFlag.SymbolicConditionals)

        dr.set_flag(dr.JitFlag.SymbolicLoops, False)
        dr.set_flag(dr.JitFlag.SymbolicCalls, False)
        dr.set_flag(dr.JitFlag.OptimizeCalls, False)
        dr.set_flag(dr.JitFlag.SymbolicConditionals, False)
    try:
        yield
    finally:
        if when:
            dr.set_flag(dr.JitFlag.SymbolicLoops, old_loop_flag)
            dr.set_flag(dr.JitFlag.SymbolicCalls, old_call_flag)
            dr.set_flag(dr.JitFlag.OptimizeCalls, old_state_flag)
            dr.set_flag(dr.JitFlag.SymbolicConditionals, old_conditional_flag)


@contextmanager
def dr_jit_context(set=True, when=True):
    if when:
        old_loop_flag = dr.flag(dr.JitFlag.SymbolicLoops)
        old_call_flag = dr.flag(dr.JitFlag.SymbolicCalls)
        old_state_flag = dr.flag(dr.JitFlag.OptimizeCalls)
        old_conditional_flag = dr.flag(dr.JitFlag.SymbolicConditionals)

        dr.set_flag(dr.JitFlag.SymbolicLoops, set)
        dr.set_flag(dr.JitFlag.SymbolicCalls, set)
        dr.set_flag(dr.JitFlag.OptimizeCalls, set)
        dr.set_flag(dr.JitFlag.SymbolicConditionals, set)
    try:
        yield
    finally:
        if when:
            dr.set_flag(dr.JitFlag.SymbolicLoops, old_loop_flag)
            dr.set_flag(dr.JitFlag.SymbolicCalls, old_call_flag)
            dr.set_flag(dr.JitFlag.OptimizeCalls, old_state_flag)
            dr.set_flag(dr.JitFlag.SymbolicConditionals, old_conditional_flag)


class TraversableDict(mi.Object):
    def __init__(self, node_dict: dict[str, tuple[mi.Object, int]]):
        super().__init__()

        self.dict_scene = {}
        for name, entry in node_dict.items():
            node, flags = entry
            self.dict_scene[name] = {"node": node, "flags": flags}

    def traverse(self, cb: mi.TraversalCallback):
        for name, entry in self.dict_scene.items():
            cb.put_object(name, entry["node"], entry["flags"])


def mitsuba_mse_loss(
    input: dr.scalar.TensorXf,
    target: dr.scalar.TensorXf,
    reduction: str = "mean",
) -> dr.scalar.TensorXf:
    squared_errors = dr.square(input - target)

    if reduction == "none":
        return squared_errors
    elif reduction == "sum":
        return dr.sum(squared_errors, axis=None)
    elif reduction == "mean":
        return dr.mean(squared_errors, axis=None)
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', 'sum'."
        )


def mitsuba_l1_loss(
    input: dr.scalar.TensorXf,
    target: dr.scalar.TensorXf,
    reduction: str = "mean",
) -> dr.scalar.TensorXf:
    absolute_errors = dr.abs(input - target)

    if reduction == "none":
        return absolute_errors
    elif reduction == "sum":
        return dr.sum(absolute_errors, axis=None)
    elif reduction == "mean":
        return dr.mean(absolute_errors, axis=None)
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', 'sum'."
        )


def _dr_smooth_l1_op(input, target, beta: float):
    diff = input - target
    abs_err = dr.abs(diff)
    sq_err = 0.5 * dr.square(diff) / beta

    res = dr.select(abs_err < beta, sq_err, abs_err - 0.5 * beta)
    return res


def mitsuba_smooth_l1_loss(
    input: dr.scalar.TensorXf,
    target: dr.scalar.TensorXf,
    reduction: str = "mean",
    beta: float = 1.0,
) -> dr.scalar.TensorXf:
    if beta == 0.0:
        return mitsuba_l1_loss(input, target, reduction=reduction)

    smooth_l1_err = _dr_smooth_l1_op(input, target, beta)

    if reduction == "none":
        return smooth_l1_err
    elif reduction == "sum":
        return dr.sum(smooth_l1_err, axis=None)
    elif reduction == "mean":
        return dr.mean(smooth_l1_err, axis=None)
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', 'sum'."
        )


def get_mitsuba_loss(type: str = "mse_loss", force_vectorized: bool = False):
    if type == "mse_loss":
        loss_fn = mitsuba_mse_loss
    elif type == "l1_loss":
        loss_fn = mitsuba_l1_loss
    elif type == "smooth_l1_loss":
        loss_fn = mitsuba_smooth_l1_loss
    else:
        raise ValueError(
            f"Unknown loss type {type}, expected one of ['mse_loss', 'l1_loss', 'smooth_l1_loss']"
        )
    if force_vectorized:

        def loss_fn_vectorized(*args, **kwargs):
            with dr_jit_context(set=True):
                return loss_fn(*args, **kwargs)

        return loss_fn_vectorized
    return loss_fn
