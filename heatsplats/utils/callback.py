from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

import heatsplats
from heatsplats.utils.typing import *

from .base import BaseObject


class Callback(BaseObject):
    r"""Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks

    Based on pytorch lightning callback
    """

    @dataclass
    class Config(BaseObject.Config):
        pass

    cfg: Config

    @property
    def state_key(self) -> str:
        """Identifier for the state of the callback.

        Used to store and retrieve a callback's state from the checkpoint dictionary by
        ``checkpoint["callbacks"][state_key]``. Implementations of a callback need to provide a unique state key if 1)
        the callback has state and 2) it is desired to maintain the state of multiple instances of that callback.

        """
        return self.__class__.__qualname__

    @property
    def _legacy_state_key(self) -> type["Callback"]:
        """State key for checkpoints saved prior to version 1.5.0."""
        return type(self)

    def _generate_state_key(self, **kwargs: Any) -> str:
        """Formats a set of key-value pairs into a state key string with the callback class name prefixed. Useful for
        defining a :attr:`state_key`.

        Args:
            **kwargs: A set of key-value pairs. Must be serializable to :class:`str`.

        """
        return f"{self.__class__.__qualname__}{repr(kwargs)}"

    # Add anything needed below

    def on_after_configure(self, obj, *args, **kwargs) -> None:
        """Called after ``obj.configure()`` in ``obj.__init__()`` and when a callback is added."""

    def on_before_backward(self, trainer, loss: Tensor) -> None:
        """Called before ``loss.backward()``, only in trainers."""

    def on_after_backward(self, trainer) -> None:
        """Called after ``loss.backward()`` and before optimizers are stepped, only in trainers."""

    def on_before_optimizer_step(
        self,
        trainer,
        optimizer: Optimizer,
    ) -> None:
        """Called before ``optimizer.step()``, only in trainers."""

    def on_before_zero_grad(
        self,
        trainer,
        optimizer: Optimizer,
    ) -> None:
        """Called before ``optimizer.zero_grad()``, only in trainers."""


@dataclass
class CallbackConfig:
    type: str = ""
    props: dict = field(default_factory=dict)


class ObjectWithCallbacks(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        callbacks: dict[str, CallbackConfig] = field(default_factory=dict)

    cfg: Config
    callbacks: dict[str, Callback]

    def _configure(self, *args, **kwargs):
        self._init_callbacks()
        super()._configure(*args, **kwargs)
        self.foreach_callback(lambda cb: cb.on_after_configure(*args, **kwargs))

    def _init_callbacks(self, *args, **kwargs):
        self.callbacks = dict()
        for name, callback in self.cfg.callbacks.items():
            self.callbacks[name] = heatsplats.find(callback.type)(
                callback.props, *args, **kwargs
            )

    def foreach_callback(
        self, fn: Callable[Concatenate[Callback, ...], None], *args, **kwargs
    ):
        for name, callback in self.callbacks.items():
            fn(callback, *args, **kwargs)
