import os
import trimesh
import argparse

import torch

from IPython import get_ipython

from optimisation import main
from heatsplats.data import MeshSamplerDataModule
from heatsplats.trainers import BaseTrainer
from heatsplats.utils import big_trimesh_pcl, show_video

from heatsplats.utils import repr_patches

__all__ = ["repr_patches", "show_video"]


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
        "rendering_config": "configs/rendering.yaml",
        "gpu": "0",
        "verbose": False,
    }
    extras_dict = {
        "data.mesh_path": "../objects/spot/spot_triangulated.obj",
        # "data.mesh_path": "../objects/bob/bob_tri.obj",
        "trainer.tracer.debug": True,
        "trainer.tracer.n_debug_traces": 100,
        "optim.iters": 20_000,
        "data.batch_size": 1024,
        # "data.sample_all_vertices": False,
        "trainer.model.n_sources": 512,
        # "trainer.model.kernel_dim": 3,
        "trainer.model.out_net": False,
        "trainer.model.normalize_colours": False,
        "data.sampling_method": "uniform",
        "renderer.point_batching": 1024,
    }

    args = argparse.Namespace(**args_dict)
    extras = [f"{k}={v}" for k, v in extras_dict.items()]
    out = main(args, extras)

    # For interactive viewer
    optimisation: BaseTrainer = out["optimisation"]
    datamodule: MeshSamplerDataModule = out["datamodule"]
    mesh: trimesh.Trimesh = datamodule.mesh
    v_colours, gt_colours, init_colours = out["colours"]

    v_colours = (v_colours.clamp(min=0, max=1) * 255).to(dtype=torch.uint8)
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

    gt_renderings, result_renderings, combined_renderings = out["renderings"]

    print("You can now visualise the followings:")
    print("  - Initial mesh: init_mesh.show()")
    print("  - GT mesh: gt_mesh.show()")
    print("  - Vert coloured optimised mesh: v_mesh.show()")
    print("  - Vert coloured optimised mesh with final kernel pos: v_scene.show()")
    print("  - Vert coloured optimised mesh with traces: v_scene_traces.show()")
    print("  - GT renderings: show_video(gt_renderings)")
    print("  - Optimised mesh renderings: show_video(result_renderings)")
    print("  - Combined renderings: show_video(combined_renderings)")
