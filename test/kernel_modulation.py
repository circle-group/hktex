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

from heatsplats.utils import (
    load_mesh,
    heat_diffusion,
    normalise_colours,
    soft_step,
    rescaled_soft_step,
    combine_videos,
    show_video,
    combine_images,
    show_image,
)
from heatsplats.utils import repr_patches

from heatsplats.rendering.heat_kernels_renderer import HeatKernelsRenderer
from heatsplats.rendering.vertex_colours_renderer import VertexColoursRenderer
from heatsplats.modules import Mesh, Model, EigenAlboInterpolation


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
    }
    eigalbo_config = {
        "k_eig": 256,
        "use_precomputed": True,
        # "precompute_anisotropies": [1, 10, 50, 100],
        # "precompute_angles_every_deg": 5,
        "mesh_path": fname,
        # "precomputed_name": "eigen_albo",
        "distance_weighting": "none",  # "gaussian_0.5",
    }
    device = "cuda:0"

    tri_mesh = load_mesh(fname, merge_tex=True, bake_vert_colors=True)
    # tri_mesh = load_mesh(fname, merge_tex=False, bake_vert_colors=False)
    our_mesh = Mesh.from_trimesh(tri_mesh, device=device)
    model = Model(model_cfg, our_mesh)
    eigalbo_interp = EigenAlboInterpolation(eigalbo_config, our_mesh)

    # model._diff_times = torch.nn.Parameter(torch.tensor([0.5, 0.8], device=device))
    model._angles = torch.nn.Parameter(
        torch.deg2rad(torch.tensor([45.0, 10.0], device=device))
    )
    model._anisotropies = torch.nn.Parameter(
        torch.log(torch.tensor([98, 52], device=device))
    )
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
        torch.tensor([20, 20], dtype=torch.float, device=device)
    )
    model._thresholds = torch.nn.Parameter(
        torch.tensor([0.2, 0.999], dtype=torch.float, device=device)
    )
    model._kernel_face_ids = torch.tensor([2000, 4682], device=device)

    # model._diff_times = torch.nn.Parameter(model._diff_times / 10)
    # model._diff_times = torch.nn.Parameter(model._diff_times.clamp(max=0.5))

    # model._kernel_colours = torch.nn.Parameter(model._kernel_colours / 3)

    # print("model.angles", model.angles)
    # print("model.anisotropies", model.anisotropies)
    # print("model._diff_times", model._diff_times)
    # print("model.diff_times", model.diff_times)

    # Vertex colours ###################################################################
    B, V = model.N_sources, our_mesh.N_verts
    # v_colours = torch.zeros([B, V, model.kernel_dim], device=device)
    v_colours = torch.zeros([B, V, 1], device=device)

    albo_weights = eigalbo_interp.interpolate_anisotropies(
        angles=model.angles, scales=model.anisotropies
    )
    albo_evals, albo_evecs, mass = eigalbo_interp.albo_vertices(
        albo_weights=albo_weights
    )

    kernel_vert_idx = our_mesh.get_face_vertices(model.kernel_face_ids)
    barycentric_coords = our_mesh.cartesian_to_barycentric(
        model.kernel_locations, kernel_vert_idx
    )

    kernel_evecs, kernel_mass = eigalbo_interp.barycentric_albo_gaussians(
        albo_weights=albo_weights,
        barycentric_coords=barycentric_coords,
        vert_idx=kernel_vert_idx,
    )

    # v_colours = torch.cat((v_colours, model.kernel_colours.unsqueeze(1)), dim=1)
    v_colours = torch.cat((v_colours, torch.ones([B, 1, 1], device=device)), dim=1)
    albo_evecs = torch.cat(((albo_evecs, kernel_evecs.unsqueeze(1))), dim=1)
    mass = torch.cat((mass.expand(B, -1), kernel_mass.unsqueeze(-1)), dim=1)

    # v_colours = heat_diffusion(
    #     v_colours, mass, albo_evals, albo_evecs, model.diff_times
    # ).sum(dim=0)
    # v_colours = v_colours[:V]

    def stiefel_projx(x: torch.Tensor) -> torch.Tensor:
        U, _, V = torch.linalg.svd(x, full_matrices=False)
        return torch.einsum("...ik,...kj->...ij", U, V)

    # albo_evecs = stiefel_projx(albo_evecs)

    v_colours = heat_diffusion(
        v_colours, mass, albo_evals, albo_evecs, model.diff_times
    )
    v_colours = v_colours / (v_colours[:, V, :].unsqueeze(1) + 1e-8)
    v_colours = v_colours[:, :V, :]

    v_colours = rescaled_soft_step(
        v_colours, epsilon=model.thresholds, sharpness=model.sharpnesses
    )

    v_colours = v_colours * model.opacities.view(-1, 1, 1)
    v_colours = v_colours * model.kernel_colours.unsqueeze(1)
    v_colours = v_colours.sum(dim=0)

    if model_cfg["normalize_colours"]:
        v_colours = normalise_colours(v_colours)

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

    # Change kernel angles
    hk_renderer = HeatKernelsRenderer({"camera_config": {"azimuth_deg": -90}})
    vc_renderer = VertexColoursRenderer({"camera_config": {"azimuth_deg": -90}})
    # hk_renderer = HeatKernelsRenderer({"camera_config": {"azimuth_deg": 0}})
    # vc_renderer = VertexColoursRenderer({"camera_config": {"azimuth_deg": 0}})
    model._angle_scale = 1
    model._angle_offset = 0

    hk_renderer.mega_kernel(False)
    images = []
    images_vertices = []
    for angle in [0, 10, 20, 30, 60, 90, 120, 150]:
        model._angles = torch.nn.Parameter(
            torch.deg2rad(torch.ones_like(model.angles, device=device) * angle)
        )
        mi_mesh = hk_renderer.mesh_to_mitsuba(tri_mesh, our_mesh, model, eigalbo_interp)
        images.append(hk_renderer.render(mi_mesh, denoise=True))
        v_colours = torch.zeros([B, V, 1], device=device)

        albo_weights = eigalbo_interp.interpolate_anisotropies(
            angles=model.angles, scales=model.anisotropies
        )
        albo_evals, albo_evecs, mass = eigalbo_interp.albo_vertices(
            albo_weights=albo_weights
        )

        kernel_vert_idx = our_mesh.get_face_vertices(model.kernel_face_ids)
        barycentric_coords = our_mesh.cartesian_to_barycentric(
            model.kernel_locations, kernel_vert_idx
        )

        kernel_evecs, kernel_mass = eigalbo_interp.barycentric_albo_gaussians(
            albo_weights=albo_weights,
            barycentric_coords=barycentric_coords,
            vert_idx=kernel_vert_idx,
        )

        v_colours = torch.cat((v_colours, torch.ones([B, 1, 1], device=device)), dim=1)
        albo_evecs = torch.cat(((albo_evecs, kernel_evecs.unsqueeze(1))), dim=1)
        mass = torch.cat((mass.expand(B, -1), kernel_mass.unsqueeze(-1)), dim=1)
        v_colours = heat_diffusion(
            v_colours, mass, albo_evals, albo_evecs, model.diff_times
        )
        v_colours = v_colours / (v_colours[:, V, :].unsqueeze(1) + 1e-8)
        v_colours = v_colours[:, :V, :]

        v_colours = rescaled_soft_step(
            v_colours, epsilon=model.thresholds, sharpness=model.sharpnesses
        )

        v_colours = v_colours * model.opacities.view(-1, 1, 1)
        v_colours = v_colours * model.kernel_colours.unsqueeze(1)
        v_colours = v_colours.sum(dim=0)
        if model_cfg["normalize_colours"]:
            v_colours = normalise_colours(v_colours)

        out_mesh = tri_mesh.copy()
        out_mesh.visual = trimesh.visual.ColorVisuals(
            out_mesh, vertex_colors=v_colours.cpu().detach().numpy()
        )
        mi_mesh_2 = vc_renderer.mesh_to_mitsuba(out_mesh)
        images_vertices.append(vc_renderer.render(mi_mesh_2, denoise=True))

    hk_renderer.flush_cache()

    combined_image = combine_images(*images)
    combined_image_vertices = combine_images(*images_vertices)

    print(f"show all angles with: show_image(combined_image)")
