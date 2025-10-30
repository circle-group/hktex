from contextlib import contextmanager

import drjit as dr
import mitsuba as mi

__all__ = ["dr_no_jit"]


# Trick from:  https://github.com/gerwang/diff-sdf/blob/9122375e37ca8757f77cd3fdc124e36b46c523be/python/util.py#L228
@contextmanager
def dr_no_jit(when=True):
    if when:
        old_loop_flag = dr.flag(dr.JitFlag.SymbolicLoops)
        old_call_flag = dr.flag(dr.JitFlag.SymbolicCalls)
        old_state_flag = dr.flag(dr.JitFlag.OptimizeCalls)

        dr.set_flag(dr.JitFlag.SymbolicLoops, False)
        dr.set_flag(dr.JitFlag.SymbolicCalls, False)
        dr.set_flag(dr.JitFlag.OptimizeCalls, False)
    try:
        yield
    finally:
        if when:
            dr.set_flag(dr.JitFlag.SymbolicLoops, old_loop_flag)
            dr.set_flag(dr.JitFlag.SymbolicCalls, old_call_flag)
            dr.set_flag(dr.JitFlag.OptimizeCalls, old_state_flag)
