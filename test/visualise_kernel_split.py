import sys
from pathlib import Path


try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import torch
from functools import partial
from omegaconf import OmegaConf

from heatsplats.utils import (
    load_mesh,
    combine_videos,
    show_video,
    combine_images,
    show_image,
)
from heatsplats.utils import repr_patches, box_border

from heatsplats.rendering.heat_kernels_renderer import HeatKernelsRenderer
from heatsplats.modules import Mesh, Model, EigenAlboInterpolation
from heatsplats.density_controllers import BaseDensityController
from heatsplats.trainers import parse_optimizers
from heatsplats.modules import CPUGeodesicTracer


if __name__ == "__main__":

    fname = "../objects/spot/spot_triangulated.obj"
    model_cfg = {
        "weights": None,
        "n_sources": 2,
        "out_dim": 3,
        "kernel_dim": 3,
        "out_net": False,
        "normalize_colours": False,
    }
    eigalbo_config = {
        "k_eig": 256,
        "use_precomputed": True,
        "precompute_anisotropies": [5, 15, 30, 60, 100],
        "precompute_angles_every_deg": 30,
        "mesh_path": fname,
        "precomputed_name": "eigen_albo",
    }
    device = "cuda:0"

    tri_mesh = load_mesh(fname, merge_tex=True, bake_vert_colors=True)
    our_mesh = Mesh.from_trimesh(tri_mesh, device=device)
    model = Model(model_cfg, our_mesh)
    eigalbo_interp = EigenAlboInterpolation(eigalbo_config, our_mesh)

    # Void all activations for interpretability over ease of optimisation ##############
    model._angle_act = lambda x: torch.deg2rad(x)
    model._anis_act = lambda x: x
    model._opacity_act = lambda x: x
    model._sharpness_act = lambda x: x
    model._thresholds_act = lambda x: x
    model._colour_act = lambda x: x

    # Manually set parameters ##########################################################
    model._angles = torch.nn.Parameter(
        torch.deg2rad(torch.tensor([0.0, 0.0], device=device))
    )
    model._anisotropies = torch.nn.Parameter(torch.tensor([30.0, 30.0], device=device))
    model._kernel_colours = torch.nn.Parameter(
        torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], dtype=torch.float, device=device)
    )
    model._kernel_locations = torch.nn.Parameter(
        torch.tensor(
            [[-0.3444, -0.5293, -0.0918], [-0.3068, 0.0106, 0.7313]], device=device
        )
    )
    model._opacities = torch.nn.Parameter(
        torch.tensor([1.0, 1.0], dtype=torch.float, device=device)
    )
    model._sharpnesses = torch.nn.Parameter(
        torch.tensor([100.0, 100.0], dtype=torch.float, device=device)
    )
    model._thresholds = torch.nn.Parameter(
        torch.tensor([0.5, 0.5], dtype=torch.float, device=device)
    )
    model._kernel_face_ids = torch.tensor([2000, 4682], device=device)

    # Render as rings
    model.kernel_filter_func = partial(box_border, thickness=0.03)
    ####################################################################################

    hk_renderer = HeatKernelsRenderer({"camera_config": {"azimuth_deg": -90}})

    hk_renderer.mega_kernel(False)
    mi_mesh = hk_renderer.mesh_to_mitsuba(tri_mesh, our_mesh, model, eigalbo_interp)
    video = hk_renderer.rotating_video(mi_mesh, 5)
    hk_renderer.flush_cache()

    # Split both kernels ###############################################################

    cfg_optim = OmegaConf.create(
        [
            {
                "name": "Adam",
                "args": {"lr": 0},
                "params": {
                    "_kernel_colours": {},
                    "_angles": {},
                    "_anisotropies": {},
                    "_thresholds": {},
                    "_sharpnesses": {},
                    "_opacities": {},
                    "_kernel_locations": {},  # just a placeholder, should be a GeodesicOpt
                },
            }
        ]
    )
    optimizers = parse_optimizers(cfg_optim, model)

    tracer = CPUGeodesicTracer({}, our_mesh)
    density_controller = BaseDensityController(
        {}, mesh=our_mesh, model=model, optimizers=optimizers
    )

    albo_weights = eigalbo_interp.interpolate_anisotropies(
        angles=model.angles, scales=model.anisotropies
    )

    kernel_info = model.prepare_kernels_for_diffusion(
        mesh=our_mesh,
        eigalbo_interp=eigalbo_interp,
        albo_weights=albo_weights,
    )

    density_controller.split(
        torch.tensor([True, True]), eigalbo_interp, kernel_info, tracer
    )

    model._kernel_colours = torch.nn.Parameter(torch.randn_like(model._kernel_colours))

    hk_renderer.mega_kernel(False)
    mi_mesh_2 = hk_renderer.mesh_to_mitsuba(tri_mesh, our_mesh, model, eigalbo_interp)
    video_2 = hk_renderer.rotating_video(mi_mesh_2, 5)
    hk_renderer.flush_cache()

    combined_video = combine_videos(video, video_2)
    print("Show video of split kernels: show_video(combined_video)")
