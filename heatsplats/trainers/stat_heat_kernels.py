from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from termcolor import colored

import torch
import torch.nn as nn
import torch.nn.functional as F

import heatsplats
from heatsplats.data import MeshSamplerDataModule
from heatsplats.utils.typing import *

from .base import BaseTrainer


@heatsplats.register("trainers.stationary-heat-kernels")
class StationaryHeatKernelsTrainer(BaseTrainer):
    @dataclass
    class Config(BaseTrainer.Config):
        gt_source_sampling_method: Optional[str] = "fps"

    cfg: Config

    def configure(
        self,
        datamodule: MeshSamplerDataModule,
        **kwargs,
    ):
        self.cfg.kernel_dim = 3
        self.cfg.lrs.out_net = 0

        super().configure(datamodule, **kwargs)

        assert hasattr(
            datamodule, "bake_heat"
        ), "OptimiseHeatKernelsToKnownStationary requires a datamodule with configure and bake_heat"
        datamodule.configure(
            n_sources=self.n_sources,
            diff_time_scaler_func=self._diff_time_scaler_func,
        )
        datamodule.bake_heat(
            self.eigalbo_interp, self.normalize_colours, device=self.device
        )

    @property
    def gt_splats(self):
        return self.datamodule.gt_splats

    @property
    def _errors(self):
        angles_error = (
            (self.angles.cpu() - torch.deg2rad(self.gt_splats["angles"]))
            .pow(2)
            .sum()
            .pow(0.5)
        )
        anisotropies_error = (
            (self.anisotropies.cpu() - self.gt_splats["anisotropies"])
            .pow(2)
            .sum()
            .pow(0.5)
        )
        diff_times_error = (
            (self.diff_times.cpu() - self.gt_splats["diff_times"]).pow(2).sum().pow(0.5)
        )
        kernel_colours_error = (
            (self.kernel_colours.cpu() - self.gt_splats["kernel_colours"])
            .pow(2)
            .sum()
            .pow(0.5)
        )
        return {
            "printables": (
                "ERRORS: "
                + colored(f"Angles: {angles_error}, ", "yellow")
                + colored(f"Anisotropies: {anisotropies_error}, ", "green")
                + colored(f"Diff times: {diff_times_error}, ", "blue")
                + colored(f"Kernel colours: {kernel_colours_error}", "red")
            ),
            "angles": angles_error,
            "anisotropies": anisotropies_error,
            "diff_times": diff_times_error,
            "kernel_colours": kernel_colours_error,
        }

    @staticmethod
    def plot_errors(errors_lists):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(errors_lists["angles"], label="Angles Error")
        axs[0, 0].set_title("Angles Error")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Error")

        axs[0, 1].plot(errors_lists["anisotropies"], label="Anisotropies Error")
        axs[0, 1].set_title("Anisotropies Error")
        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Error")

        axs[1, 0].plot(errors_lists["diff_times"], label="Diff Times Error")
        axs[1, 0].set_title("Diff Times Error")
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Error")

        axs[1, 1].plot(errors_lists["kernel_colours"], label="Kernel Colours Error")
        axs[1, 1].set_title("Kernel Colours Error")
        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_ylabel("Error")

        for ax in axs.flat:
            ax.legend()

        plt.tight_layout()
        plt.show()
