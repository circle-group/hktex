import torch
import torch.nn as nn
import drjit as dr
import mitsuba as mi
import numpy as np

from dataclasses import dataclass

from hktex.utils.typing import *


class TorchTexture(mi.Texture):
    def __init__(self, props: mi.Properties) -> None:
        mi.Texture.__init__(self, props)
        self.network = None

    def traverse(self, callback):
        if self.network is not None:
            self.network.traverse(callback)
        callback.put("texture", self, mi.ParamFlags.NonDifferentiable)

    def eval(self, si, active=True, dirs=None, norms=None, albedo=None):
        return self.network.eval(si, dirs, norms, albedo)

    def eval_1(self, si, active=True):
        raise NotImplementedError()

    def eval_1_grad(self, *args, **kwargs):
        raise NotImplementedError()

    def eval_3(self, *args, **kwargs):
        raise NotImplementedError()

    def mean(self, *args, **kwargs):
        raise NotImplementedError()

    def to_string(self):
        return "TorchTexture[\n" f"  network={self.network}\n" "]"


mi.register_texture("torch_texture", TorchTexture)


def vec_to_tens_safe(vec):
    # A utility function that converts a Vector3f to a TensorXf safely in mitsuba while keeping the gradients;
    # a regular type cast mi.TensorXf(vector) detaches the gradients
    return mi.TensorXf(dr.ravel(vec), shape=(dr.shape(vec)[1], dr.shape(vec)[0]))


class MitsubaWrapper(nn.Module):
    def __init__(self, name: str = None):
        super().__init__()
        self.grad_activator = mi.Vector3f(0)
        self.name = name or type(self).__name__

    def eval(self, si, dirs=None, norms=None, albedo=None):
        result = self._eval(si, dirs, norms, albedo)
        return result

    def traverse(self, callback):
        callback.put(
            "grad_activator", self.grad_activator, mi.ParamFlags.Differentiable
        )
        self._traverse(callback)

    def _eval(self, si, dirs, norms, albedo):
        raise NotImplementedError()

    def _traverse(self, callback):
        pass

    def to_string(self):
        return f"MitsubaWrapper[name='{self.name}']"
