import trimesh
import argparse

import torch

from optimisation import main, OptimiseHeatKernels
from heatsplats.utils import big_trimesh_pcl

if __name__ == "__main__":
    args_dict = {
        "config": "configs/vertex_colour_texture_fitting.yaml",
        "gpu": "0",
        "verbose": True,
    }
    extras_dict = {"trainer.tracer.debug": True, "optim.iters": 500}

    args = argparse.Namespace(**args_dict)
    extras = [f"{k}={v}" for k, v in extras_dict.items()]
    out = main(args, extras)

    # For interactive viewer
    optimisation: OptimiseHeatKernels = out["optimisation"]
    mesh: trimesh.Trimesh = out["mesh"]
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
