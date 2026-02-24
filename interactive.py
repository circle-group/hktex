import os
import yaml

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import trimesh
import argparse

import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

import torch

from IPython import get_ipython
from omegaconf import OmegaConf
from optimisation import main
from heatsplats.data import MeshSamplerDataModule
from heatsplats.trainers import BaseTrainer
from heatsplats.utils import (
    big_trimesh_pcl,
    show_video,
    mibitmaps2torch,
    compute_all_image_metrics,
)

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

# dr.set_flag(dr.JitFlag.Debug, True)


if __name__ == "__main__":
    os.environ["CUDA_HOME"] = "/vol/cuda/12.2.0/"
    args_dict = {
        # "config": "configs/vertex_colour_texture_fitting.yaml",
        # "config": "configs/known_vertex_colour_fitting.yaml",
        # "config": "configs/uv_texture_fitting.yaml",
        "config": "configs/uv_texture_fitting_knn.yaml",
        "rendering_config": "configs/rendering.yaml",
        "gpu": "0",
        "verbose": True,
    }
    extras_dict = {
        # "data.mesh_path": "/data2/objaverse/hf-objaverse-v1/glbs/000-087/0e708d1e0ce0447ba5637a5320f5729c.glb",
        # "data.mesh_path": "/data2/objaverse/hf-objaverse-v1/glbs/000-018/998d641ce1c74e44978a91fedc849905.glb",
        # "data.mesh_path": "/data2/objaverse/hf-objaverse-v1/glbs/000-074/5ecf9d1175ae405a9a073db305786411.glb",
        # "data.mesh_path": "/data2/objaverse/hf-objaverse-v1/glbs/000-101/818e088dc59f4a89bfea14cb46a4beca.glb",
        # "data.mesh_path": "/data2/objaverse/hf-objaverse-v1/glbs/000-138/6713cc0cdad34f89a0256c5d2f68b7c1.glb",
        # "data.mesh_path": "/data2/objaverse/hf-objaverse-v1/glbs/000-013/d79a32a512c64c5e93dc856864789a7e.glb",
        "data.mesh_path": "/data2/objaverse/hf-objaverse-v1/glbs/000-096/db5f9c28708142909b15212625a127f9.glb",
        # "data.mesh_path": "/data2/objaverse/hf-objaverse-v1/glbs/000-066/e7caba92073d4adba3477c21aa25e91f.glb",
        # "data.mesh_path": "../objects/spot/spot_triangulated.obj",
        # "data.mesh_path": "../objects/bob/bob_tri.obj",
        # "data.mesh_path": "../objects/human_tri/RUST_3d_Low1.obj",
        # "data.mesh_path": "../objects/cat_tri/12221_Cat_v1_l3.obj",
        "trainer.tracer.debug": False,
        # "trainer.tracer.n_debug_traces": 100,
        "optim.iters": 1000,
        # "data.batch_size": 512,
        # "data.sample_all_vertices": False,
        # "trainer.model.n_sources": 500,
        # "trainer.model.kernel_dim": 3,
        # "trainer.model.out_net": False,
        # "trainer.model.normalize_colours": False,
        # "data.sampling_method": "uniform",
        "renderer.point_batching": 512,
        "renderer.n_rotating_frames": 3,
        "renderer.integrator_config.type": "prb",
        "renderer.integrator_config.meta.max_depth": 2,
        "trainer.renderer_mega_kernel": False,
        "renderer.camera_config.tile_size_heatkernels": None,
        # "renderer.camera_config.camera_distance": 2.2,
        # "trainer.eigen_albo.local_frames": "axis_aligned_20",
        # "trainer.model.init_min_threshold": 0.9999,
        # "trainer.density_controllers": [],
        "trainer.model.range_enforcement_type": "pgd",
        "trainer.model.init_kernel_edge_type": "uniform",
        "trainer.eigen_albo.use_euclidian_distance": False,
        "trainer.eigen_albo.faiss.index_type": "flat",
    }

    args = argparse.Namespace(**args_dict)

    # Overrides elements in the lists within config
    base_cfg = load_config(args.config)
    density_controllers_list = base_cfg.trainer.density_controllers
    # density_controllers_list[1].args.max_kernels = 5_000
    extras_dict["trainer.density_controllers"] = yaml.dump(
        OmegaConf.to_container(density_controllers_list)
    )

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
    if render:
        gt_rend, result_rend, ring_rend, combined_rend = out["renderings"]

    gt = mibitmaps2torch(gt_rend)
    res = mibitmaps2torch(result_rend)
    metrics = compute_all_image_metrics(res, gt)
    print("Metrics:", metrics)

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


def temp():
    gt_1 = optimisation.render_gt_raw()

    import numpy as np
    import torch.nn.functional as F

    renderer = DifferentiableHeatKernelsRenderer(optimisation.cfg.renderer)
    renderer.mega_kernel(False)

    mi_mesh, mi_texture = renderer.mesh_to_mitsuba(
        optimisation.datamodule.mesh,
        optimisation.mesh,
        optimisation.model,
        optimisation.eigalbo_interp,
    )

    scene = renderer.make_scene(mi_mesh, with_params=False)

    params = mi.traverse(mi_texture)
    dr.enable_grad(params["grad_activator"])
    print(params)
    print("-----------")

    print(optimisation.model._angles)

    img = mi.render(scene, params=params, seed=0, seed_grad=0 + 1)
    loss = dr.mean((img - gt_1) ** 2)
    print("img ===", img)
    print("loss ===", loss)
    print("gt_1 ===", gt_1)
    dr.backward(loss)

    print("============")
    print("others:")
    for n, t in optimisation.model.named_parameters():
        if t.grad is not None:
            print(n, t)
