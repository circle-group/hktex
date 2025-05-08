import sys
from pathlib import Path
import os


sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt


from heatsplats.utils import load_config, ExperimentConfig
import heatsplats
from heatsplats.data import MeshSamplerDataModule
from heatsplats.trainers import BaseTrainer
from heatsplats.utils.typing import *


if __name__ == "__main__":

    experiment = "outputs/uv-texture-fitting/spot_triangulated@20250429-180141"

    cfg_path = os.path.join(experiment, "configs/parsed.yaml")
    cfg: ExperimentConfig = load_config(cfg_path)

    datamodule: MeshSamplerDataModule = heatsplats.find(cfg.data_type)(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")

    trainer: BaseTrainer = heatsplats.find(cfg.trainer_type)(cfg.trainer, datamodule)

    ckpt_name = os.path.join(cfg.trial_dir, "ckpts", cfg.optim.save_model_name)
    trainer.model.load_torch(ckpt_name)
    correct_v_colours = trainer.compute_vertex_colours()

    # Find interesting kernels
    print(
        f"diff times = [{trainer.model.diff_times.min()},"
        f"{trainer.model.diff_times.max()}] -> ",
        f"[{trainer.model.diff_times.argmin()},{trainer.model.diff_times.argmax()}]",
    )
    print(
        f"anisotropies = [{trainer.model.anisotropies.min()},"
        f"{trainer.model.anisotropies.max()}] -> ",
        f"[{trainer.model.anisotropies.argmin()},{trainer.model.anisotropies.argmax()}]",
    )
    print(
        f"angles = [{trainer.model.angles.min()}," f"{trainer.model.angles.max()}] -> ",
        f"[{trainer.model.angles.argmin()},{trainer.model.angles.argmax()}]",
    )

    # Perturb selected kernel and get new colours
    kernel_idx = 34
    print(f"Perturbing kernel {kernel_idx}")

    with torch.no_grad():
        trainer.model._kernel_colours[kernel_idx, :] = torch.zeros_like(
            trainer.model._kernel_colours[kernel_idx, :]
        )
    perturbed_v_colours = trainer.compute_vertex_colours()

    v_colours = (correct_v_colours - perturbed_v_colours).abs().mean(dim=1)
    v_colours_normalized = (v_colours - v_colours.min()) / (
        v_colours.max() - v_colours.min()
    )

    cmap = plt.get_cmap("plasma")
    v_colours = cmap(v_colours_normalized.detach().cpu().numpy())[:, :3]  # Use RGB only
    v_colours = (v_colours * 255).astype(np.uint8)

    v_mesh = datamodule.mesh.copy()
    v_mesh.visual = trimesh.visual.ColorVisuals(v_mesh, vertex_colors=v_colours)
