import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import trimesh
import argparse

import torch

from IPython import get_ipython

from optimisation import main
from heatsplats.data import MeshSamplerDataModule
from heatsplats.trainers import BaseTrainer
from heatsplats.utils import big_trimesh_pcl

try:
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":  # Jupyter Notebook
        print("Running in a Jupyter Notebook. Enabling autoreload...")
        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")
except NameError:
    # get_ipython() is not defined, so not running in an IPython environment
    pass


if __name__ == "__main__":
    args_dict = {
        # "config": "configs/vertex_colour_texture_fitting.yaml",
        # "config": "configs/known_vertex_colour_fitting.yaml",
        "config": "configs/uv_texture_fitting.yaml",
        "gpu": "0",
        "verbose": False,
    }
    extras_dict = {
        "data.mesh_path": "../objects/bob/bob_tri.obj",
        "trainer.tracer.debug": True,
        # "trainer.tracer.n_debug_traces": 400,
        "optim.iters": 500,
        "data.batch_size": 1024,
        # "data.sample_all_vertices": False,
        "trainer.model.n_sources": 400,
        # "trainer.model.kernel_dim": 3,
        # "trainer.model.out_net": False,
        # "trainer.model.normalize_colours": False,
    }

    args = argparse.Namespace(**args_dict)
    extras = [f"{k}={v}" for k, v in extras_dict.items()]
    out = main(args, extras)

    # For interactive viewer
    optimisation: BaseTrainer = out["optimisation"]
    datamodule: MeshSamplerDataModule = out["datamodule"]
    mesh: trimesh.Trimesh = datamodule.mesh
    v_colours, gt_colours, init_colours = out["colours"]

    v_colours = (v_colours * 255).to(dtype=torch.uint8)
    v_colours = v_colours.squeeze().detach().cpu().numpy()

    gt_colours = (gt_colours * 255).to(dtype=torch.uint8)
    gt_colours = gt_colours.squeeze().detach().cpu().numpy()

    init_colours = (init_colours * 255).to(dtype=torch.uint8)
    init_colours = init_colours.squeeze().detach().cpu().numpy()

    gt_mesh = mesh.copy()
    gt_mesh.visual = trimesh.visual.ColorVisuals(gt_mesh, vertex_colors=gt_colours)

    v_mesh = mesh.copy()
    v_mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=v_colours)
    v_scene = trimesh.Scene([v_mesh, big_trimesh_pcl(optimisation.kernel_centres)])
    if optimisation.tracer.debug:
        v_scene_traces = trimesh.Scene([v_mesh, *optimisation.debug_trimesh_traces])

    init_mesh = mesh.copy()
    init_mesh.visual = trimesh.visual.ColorVisuals(
        init_mesh, vertex_colors=init_colours
    )

    print(f"You can now visualise the followings:")
    print(f"  - Initial mesh: init_mesh.show()")
    print(f"  - GT mesh: gt_mesh.show()")
    print(f"  - Optimised mesh: v_mesh.show()")
    print(f"  - Optimised mesh with final kernel positions: v_scene.show()")
    print(f"  - Optimised mesh with traces: v_scene_traces.show()")
