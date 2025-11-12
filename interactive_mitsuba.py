import os
import yaml

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import trimesh
import argparse

import torch

import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

from IPython import get_ipython
from omegaconf import OmegaConf
from optimisation import main
from heatsplats.data import MeshSamplerDataModule
from heatsplats.trainers import BaseTrainer
from heatsplats.utils import big_trimesh_pcl, show_video

from heatsplats.utils import repr_patches, load_config

from heatsplats.rendering.diffhk_renderer import DifferentiableHeatKernelsRenderer

__all__ = ["repr_patches", "show_video"]


try:
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":  # Jupyter Notebook
        print("Running in a Jupyter Notebook. Enabling autoreload...")
        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")
except NameError:
    # get_ipython() is not defined, so not running in an IPython environment
    pass

dr.set_flag(dr.JitFlag.Debug, False)

if __name__ == "__main__":
    os.environ["CUDA_HOME"] = "/vol/cuda/12.2.0/"
    args_dict = {
        "config": "configs/uv_mitsuba_fitting.yaml",
        "rendering_config": "configs/rendering.yaml",
        "gpu": "0",
        "verbose": False,
    }
    extras_dict = {
        "data.mesh_path": "../objects/spot/spot_triangulated.obj",
        "optim.iters": 500,
        "data.batch_size": 2,
        # "data.sample_all_vertices": False,
        "trainer.network.model.n_sources": 512,
        # "trainer.model.kernel_dim": 3,
        "trainer.network.model.out_net": False,
        "trainer.network.model.normalize_colours": False,
        "trainer.network.point_batching": 1024,
        "renderer.point_batching": None,
        "renderer.n_rotating_frames": 5,
        "renderer.integrator_config.type": "path",
        # "renderer.integrator_config.meta.max_depth": 2,
        "trainer.renderer_mega_kernel": False,
        "renderer.camera_config.tile_size": 64,
        "renderer.camera_config.tile_size_heatkernels": 64,
    }

    args = argparse.Namespace(**args_dict)

    extras = [f"{k}={v}" for k, v in extras_dict.items()]

    profile = False
    render = True

    if profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            # with_stack=True,
        ) as prof:
            out = main(args, extras, render=render)

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
    else:
        out = main(args, extras, render=render)

    # For interactive viewer
    optimisation: BaseTrainer = out["optimisation"]
    datamodule: MeshSamplerDataModule = out["datamodule"]
    mesh: trimesh.Trimesh = datamodule.mesh
    v_colours, gt_colours, init_colours = out["colours"]

    # v_colours = (v_colours.clamp(min=0, max=1) * 255).to(dtype=torch.uint8)
    # v_colours = v_colours.squeeze().detach().cpu().numpy()

    # gt_colours = (gt_colours * 255).to(dtype=torch.uint8)
    # gt_colours = gt_colours.squeeze().detach().cpu().numpy()

    # init_colours = (init_colours * 255).to(dtype=torch.uint8)
    # init_colours = init_colours.squeeze().detach().cpu().numpy()

    # gt_mesh = mesh.copy()
    # gt_mesh.visual = trimesh.visual.ColorVisuals(gt_mesh, vertex_colors=gt_colours)

    # v_mesh = mesh.copy()
    # v_mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=v_colours)
    # v_scene = trimesh.Scene([v_mesh, big_trimesh_pcl(optimisation.kernel_centres)])
    # if optimisation.tracer.debug:
    #    v_scene_traces = trimesh.Scene([v_mesh, *optimisation.debug_trimesh_traces])

    # init_mesh = mesh.copy()
    # init_mesh.visual = trimesh.visual.ColorVisuals(
    #     init_mesh, vertex_colors=init_colours
    # )
    if render:
        gt_rend, result_rend, ring_rend, combined_rend = out["renderings"]

    print("You can now visualise the followings:")
    print("  - Initial mesh: init_mesh.show()")
    print("  - GT mesh: gt_mesh.show()")
    print("  - Vert coloured optimised mesh: v_mesh.show()")
    print("  - Vert coloured optimised mesh with final kernel pos: v_scene.show()")
    print("  - Vert coloured optimised mesh with traces: v_scene_traces.show()")
    print("  - GT renderings: show_video(gt_rend)")
    print("  - Optimised mesh renderings: show_video(result_rend)")
    print("  - Kernel ring renderings: show_video(ring_rend)")
    print("  - Combined renderings: show_video(combined_rend)")
