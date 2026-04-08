import sys
from pathlib import Path
import os

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import torch
import trimesh
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

from heatsplats.utils import (
    load_mesh,
    combine_videos,
    show_video,
    combine_images,
    show_image,
)
from heatsplats.utils import repr_patches

from heatsplats.rendering.heat_kernels_renderer import HeatKernelsRenderer
from heatsplats.rendering.vertex_colours_renderer import VertexColoursRenderer
from heatsplats.modules import Mesh, HeatKernelTexture, EigenAlboInterpolation


if __name__ == "__main__":

    fname = "../objects/spot/spot_triangulated.obj"
    # fname = "../objects/square_mesh.ply"
    model_cfg = {
        "weights": None,
        "n_sources": 2,
        "out_dim": 3,
        "kernel_dim": 3,
        "out_net": False,
        "normalize_colours": False,
        # "diff_time": 0.3,
        "mass_type": "one",
    }
    eigalbo_config = {
        "k_eig": 256,
        "use_precomputed": False,
        "precompute_anisotropies": [5, 15, 30, 60, 100],
        "precompute_angles_every_deg": 30,
        "mesh_path": fname,
        # "precomputed_name": "eigen_albo",
        "distance_weighting": "none",
        "local_frames": "principal_curvatures",  # "axis_aligned_8",
        "mass_type": model_cfg["mass_type"],
        "normalise_evals": False,
    }
    device = "cuda:0"

    # print(f"diff_time: {model_cfg['diff_time']}")

    tri_mesh = load_mesh(
        fname, merge_tex=True, bake_vert_colors=True, normalise_size=True
    )
    # tri_mesh = load_mesh(fname, merge_tex=False, bake_vert_colors=False)
    our_mesh = Mesh.from_trimesh(tri_mesh, device=device)
    model = HeatKernelTexture(model_cfg, our_mesh)
    eigalbo_interp = EigenAlboInterpolation(eigalbo_config, our_mesh)

    # Void all activations for interpretability over ease of optimisation ##############
    model._angle_act = lambda x: torch.deg2rad(x)
    model._anis_act = lambda x: x
    model._sharpness_act = lambda x: x
    model._thresholds_act = lambda x: x
    model._colour_act = lambda x: x

    # Manually set parameters ##########################################################
    model._mean_colour = torch.nn.Parameter(torch.zeros(1, 3, device=device))
    model._angles = torch.nn.Parameter(
        torch.deg2rad(torch.tensor([0.0, 0.0], device=device))
    )
    model._anisotropies = torch.nn.Parameter(torch.tensor([30.0, 30.0], device=device))
    model._kernel_colours = torch.nn.Parameter(
        torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], dtype=torch.float, device=device)
    )
    model._kernel_face_ids = torch.tensor([2000, 4682], device=device)

    model._kernel_locations = torch.nn.Parameter(
        our_mesh.barycentric_to_cartesian(
            torch.tensor([[0.5, 0.5, 0], [0.5, 0.5, 0]], device=device),
            our_mesh.get_face_vertices(model._kernel_face_ids),
        )
    )
    model._sharpnesses = torch.nn.Parameter(
        torch.tensor([100.0, 100.0], dtype=torch.float, device=device)
    )
    model._thresholds = torch.nn.Parameter(
        torch.tensor([0.5, 0.5], dtype=torch.float, device=device)
    )

    # Vertex colours ###################################################################
    v_colours = model.compute_vertex_colours(our_mesh, eigalbo_interp)

    out_mesh = tri_mesh.copy()
    out_mesh.visual = trimesh.visual.ColorVisuals(
        out_mesh, vertex_colors=v_colours.cpu().detach().numpy()
    )

    print("show mesh of diffusion performed on vertices with: out_mesh.show()")

    vertex_colours_renderer = VertexColoursRenderer(dict())
    mi_mesh_2 = vertex_colours_renderer.mesh_to_mitsuba(out_mesh)
    video_vert = vertex_colours_renderer.rotating_video(mi_mesh_2, 5)

    print("show video of diffusion performed on vertices with: show_video(video_vert)")

    # HKTex ############################################################################

    hk_renderer = HeatKernelsRenderer(dict())

    hk_renderer.mega_kernel(False)
    mi_mesh = hk_renderer.mesh_to_mitsuba(tri_mesh, our_mesh, model, eigalbo_interp)
    image = hk_renderer.render(mi_mesh, False)
    bitmap = mi.Bitmap(image).convert(srgb_gamma=True)
    video = hk_renderer.rotating_video(mi_mesh, 5)
    hk_renderer.flush_cache()

    print("show video with: show_video(video)")

    combined_video = combine_videos(video_vert, video)
    print("show combined videos (vert left) with: show_video(combined_video)")

    ####################################################################################
    # Change parameters ################################################################
    ####################################################################################
    hk_renderer = HeatKernelsRenderer({"camera_config": {"azimuth_deg": -90}})
    vc_renderer = VertexColoursRenderer({"camera_config": {"azimuth_deg": -90}})
    # hk_renderer = HeatKernelsRenderer({"camera_config": {"azimuth_deg": 0}})
    # vc_renderer = VertexColoursRenderer({"camera_config": {"azimuth_deg": 0}})
    # hk_renderer = HeatKernelsRenderer(
    #     {"camera_config": {"azimuth_deg": -90, "camera_distance": 7.0}}
    # )

    hk_renderer.mega_kernel(False)

    # Change angles ####################################################################
    images = []
    images_vertices = []
    for angle in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]:
        # for angle in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]:
        model._angles = torch.nn.Parameter(
            torch.ones_like(model.angles, device=device) * angle
        )

        mi_mesh = hk_renderer.mesh_to_mitsuba(tri_mesh, our_mesh, model, eigalbo_interp)
        images.append(hk_renderer.render(mi_mesh, denoise=True))

        v_colours = model.compute_vertex_colours(our_mesh, eigalbo_interp)
        out_mesh = tri_mesh.copy()
        out_mesh.visual = trimesh.visual.ColorVisuals(
            out_mesh, vertex_colors=v_colours.cpu().detach().numpy()
        )
        mi_mesh_2 = vc_renderer.mesh_to_mitsuba(out_mesh)
        images_vertices.append(vc_renderer.render(mi_mesh_2, denoise=True))

    model._angles = torch.nn.Parameter(
        torch.deg2rad(torch.tensor([0.0, 0.0], device=device))
    )

    combined_img_angles = combine_images(*images)
    combined_img_v_angles = combine_images(*images_vertices)

    hk_renderer.flush_cache()

    print(f"show all angles with: show_image(combined_img_angles)")

    # Change anisotropies ##############################################################
    images = []
    images_vertices = []
    # for anis in [20, 25, 30, 35, 40]:
    for anis in [1, 5, 10, 15, 20, 30, 45, 60, 80, 100, 200]:
        model._anisotropies = torch.nn.Parameter(
            torch.ones_like(model.anisotropies, device=device) * anis
        )

        mi_mesh = hk_renderer.mesh_to_mitsuba(tri_mesh, our_mesh, model, eigalbo_interp)
        images.append(hk_renderer.render(mi_mesh, denoise=True))

        v_colours = model.compute_vertex_colours(our_mesh, eigalbo_interp)
        out_mesh = tri_mesh.copy()
        out_mesh.visual = trimesh.visual.ColorVisuals(
            out_mesh, vertex_colors=v_colours.cpu().detach().numpy()
        )
        mi_mesh_2 = vc_renderer.mesh_to_mitsuba(out_mesh)
        images_vertices.append(vc_renderer.render(mi_mesh_2, denoise=True))

    model._anisotropies = torch.nn.Parameter(torch.tensor([10.0, 10.0], device=device))

    combined_img_anis = combine_images(*images)
    combined_img_v_anis = combine_images(*images_vertices)

    hk_renderer.flush_cache()
    print(f"show all anisotropies with: show_image(combined_img_anis)")

    # Change sharpnesses ###############################################################
    images = []
    images_vertices = []
    for sharp in [5, 10, 50, 100]:
        model._sharpnesses = torch.nn.Parameter(
            torch.ones_like(model.sharpnesses, device=device) * sharp
        )

        mi_mesh = hk_renderer.mesh_to_mitsuba(tri_mesh, our_mesh, model, eigalbo_interp)
        images.append(hk_renderer.render(mi_mesh, denoise=True))

        v_colours = model.compute_vertex_colours(our_mesh, eigalbo_interp)
        out_mesh = tri_mesh.copy()
        out_mesh.visual = trimesh.visual.ColorVisuals(
            out_mesh, vertex_colors=v_colours.cpu().detach().numpy()
        )
        mi_mesh_2 = vc_renderer.mesh_to_mitsuba(out_mesh)
        images_vertices.append(vc_renderer.render(mi_mesh_2, denoise=True))

    model._sharpnesses = torch.nn.Parameter(torch.tensor([100.0, 100.0], device=device))

    combined_img_sharp = combine_images(*images)
    combined_img_v_sharp = combine_images(*images_vertices)

    hk_renderer.flush_cache()
    print(f"show all sharpnesses with: show_image(combined_img_sharp)")

    # Change thresholds ################################################################
    images = []
    images_vertices = []
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        model._thresholds = torch.nn.Parameter(
            torch.ones_like(model.thresholds, device=device) * thresh
        )

        mi_mesh = hk_renderer.mesh_to_mitsuba(tri_mesh, our_mesh, model, eigalbo_interp)
        images.append(hk_renderer.render(mi_mesh, denoise=True))

        v_colours = model.compute_vertex_colours(our_mesh, eigalbo_interp)
        out_mesh = tri_mesh.copy()
        out_mesh.visual = trimesh.visual.ColorVisuals(
            out_mesh, vertex_colors=v_colours.cpu().detach().numpy()
        )
        mi_mesh_2 = vc_renderer.mesh_to_mitsuba(out_mesh)
        images_vertices.append(vc_renderer.render(mi_mesh_2, denoise=True))

    model._thresholds = torch.nn.Parameter(torch.tensor([0.5, 0.5], device=device))

    combined_img_thresh = combine_images(*images)
    combined_img_v_thresh = combine_images(*images_vertices)

    hk_renderer.flush_cache()
    print(f"show all thresholds with: show_image(combined_img_thresh)")

    print(f"to see coloured vertices, use combined_img_v_...")

    # Change sharpness and threshold ###################################################

    images_row = []
    # for sharp in [10, 20, 30, 40, 50]:
    for sharp in [10, 50, 100, 150, 200]:
        model._sharpnesses = torch.nn.Parameter(
            torch.ones_like(model.sharpnesses, device=device) * sharp
        )
        images = []
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]:
            model._thresholds = torch.nn.Parameter(
                torch.ones_like(model.thresholds, device=device) * thresh
            )

            mi_mesh = hk_renderer.mesh_to_mitsuba(
                tri_mesh, our_mesh, model, eigalbo_interp
            )
            images.append(hk_renderer.render(mi_mesh, denoise=True))

        images_row.append(combine_images(*images, as_bitmap=False))

    combined_img_thresh_sharp = combine_images(*images_row, horizontal=False)

    hk_renderer.flush_cache()
    print(
        "show all thresholds (rows) and sharpnesses (columns) with: ",
        "show_image(combined_img_thresh_sharp)",
    )
