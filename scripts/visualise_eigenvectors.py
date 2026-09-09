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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from hktex.modules import Mesh
from hktex.utils import (
    load_mesh,
    get_anisotropic_lbo,
    compute_eig_laplacian,
    compute_mesh_laplacian,
    align_eigen,
    combine_images,
    heat_diffusion,
    show_image,
)
from hktex.rendering.vertex_colours_renderer import VertexColoursRenderer
from hktex.utils import repr_patches

__all__ = ["repr_patches"]


def get_albo_evecs(mesh, angle_deg, scale, k_eig=256):
    angle = torch.deg2rad(torch.tensor(angle_deg)).item()

    lapl, mass = get_anisotropic_lbo(
        mesh.verts,
        mesh.faces.T,
        mesh.fnorms,
        rotation_angle=angle,
        anisotropy=scale,
        local_direction=None,
    )
    # lapl, mass = compute_mesh_laplacian(
    #     mesh.verts.cpu().detach().numpy(), mesh.faces.cpu().detach().numpy()
    # )

    lapl = lapl.astype(np.float32)
    mass = mass.astype(np.float32)

    evals, evecs = compute_eig_laplacian(lapl, mass, k_eig)
    return evals, evecs, mass


def render_evecs(
    mesh: trimesh.Trimesh,
    evecs: torch.Tensor,
    vc_renderer: VertexColoursRenderer,
    k_range: tuple = (10, 20),
    cmap_mode: str = "vibrant_narrow_white",
):
    images_vertices = []
    if cmap_mode == "vibrant_narrow_white":
        # Narrow white band (0.48 to 0.52), saturated key points, deep contrast endpoints
        cmap = LinearSegmentedColormap.from_list(
            "custom_evec",
            [
                (0.00, "#2b3b48"),
                (0.20, "#4b6274"),
                (0.44, "#6b89a1"),
                (0.48, "#f0f4f7"),
                (0.50, "#ffffff"),
                (0.52, "#fcf0f2"),
                (0.56, "#ef5b6c"),
                (0.80, "#df4e5c"),
                (1.00, "#b82a38"),
            ],
        )
    elif cmap_mode == "white_center_linear":
        cmap = LinearSegmentedColormap.from_list(
            "custom_evec", ["#4b6274", "#ffffff", "#df4e5c"]
        )
    else:
        cmap = plt.get_cmap("viridis")

    if isinstance(evecs, torch.Tensor):
        evecs = evecs.detach().cpu().numpy()

    for k in range(*k_range):
        evec = evecs[:, k]
        norm_evec = (evec - evec.min()) / (evec.max() - evec.min() + 1e-8)
        v_colours = (cmap(norm_evec)[:, :3] * 255).astype(np.uint8)

        out_mesh = mesh.copy()
        out_mesh.visual = trimesh.visual.ColorVisuals(out_mesh, vertex_colors=v_colours)
        mi_mesh = vc_renderer.mesh_to_mitsuba(out_mesh)
        images_vertices.append(vc_renderer.render(mi_mesh, denoise=True))

    combined_image = combine_images(*images_vertices)
    return combined_image


def render_heat_diff(t_mesh, mass, evals, evecs, device="cuda:0"):
    idxs = torch.tensor([721, 2364], device="cuda:0")  # for spot mesh

    B, V = idxs.shape[0], t_mesh.vertices.shape[0]
    diff_times = torch.tensor([1e-3] * B, device="cuda:0")

    evecs = torch.stack([torch.tensor(evecs)] * B, dim=0).to(device)
    evals = torch.stack([torch.tensor(evals)] * B, dim=0).to(device)
    mass = torch.tensor(mass).to(device)

    v_colours = torch.zeros([B, V, 1], device=device)
    v_colours[torch.arange(B), idxs, 0] = 1.0

    v_colours = heat_diffusion(
        v_colours, mass, evals, evecs, diff_times, at_vertices=True
    )

    v_colours = v_colours / (v_colours[torch.arange(B), idxs, :].unsqueeze(1) + 1e-8)
    rand_colours = torch.rand((B, 3), device=device)
    v_colours = v_colours * rand_colours.unsqueeze(1)
    v_colours = v_colours.sum(dim=0)

    out_mesh = t_mesh.copy()
    out_mesh.visual = trimesh.visual.ColorVisuals(
        out_mesh, vertex_colors=v_colours.cpu().detach().numpy()
    )
    mi_mesh = vc_renderer.mesh_to_mitsuba(out_mesh)
    return vc_renderer.render(mi_mesh, denoise=True)


if __name__ == "__main__":

    fname = "/homes/sf3018/Documents/objects/spot/spot_triangulated.obj"
    tri_mesh = load_mesh(fname, merge_tex=False, bake_vert_colors=False)

    # tri_mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)

    vc_renderer = VertexColoursRenderer(
        {
            "camera_config": {
                "azimuth_deg": -90,
                "camera_distance": 3.5,
                "img_width": 1024,
                "img_height": 1024,
            }
        }
    )
    # vc_renderer = VertexColoursRenderer({"camera_config": {"azimuth_deg": 0}})

    our_mesh = Mesh.from_trimesh(tri_mesh, device="cuda:0")
    k_ranges = (10, 20)

    evals_base, evecs_base, mass = get_albo_evecs(our_mesh, angle_deg=0, scale=10)
    img_evecs_base = render_evecs(tri_mesh, evecs_base, vc_renderer, k_ranges)

    evals_to_align, evecs_to_align, _ = get_albo_evecs(our_mesh, angle_deg=45, scale=10)
    img_evecs_to_align = render_evecs(tri_mesh, evecs_to_align, vc_renderer, k_ranges)

    evecs_aligned, _ = align_eigen(
        evecs_base, evecs_to_align, evals_to_align, mass, align_rotation=True
    )
    img_evecs_aligned = render_evecs(tri_mesh, evecs_aligned, vc_renderer, k_ranges)

    print("Copy and paste the followings to show all the eigenvectors:")
    print(
        "show_image(img_evecs_to_align)\nshow_image(img_evecs_base)\nshow_image(img_evecs_aligned)"
    )

    heat_diff_before_after = []
    set_of_images = [img_evecs_base]
    all_angles = [30, 45, 60, 90, 120]
    for a in all_angles:
        evals_to_align, evecs_to_align, _ = get_albo_evecs(
            our_mesh, angle_deg=a, scale=10
        )
        evecs_aligned, evals_aligned = align_eigen(
            evecs_base, evecs_to_align, evals_to_align, mass, align_rotation=True
        )
        set_of_images.append(
            render_evecs(tri_mesh, evecs_aligned, vc_renderer, k_ranges)
        )

        heat_diff_before = render_heat_diff(
            tri_mesh, mass, evals_to_align, evecs_to_align, device="cuda:0"
        )
        heat_diff_after = render_heat_diff(
            tri_mesh, mass, evals_aligned, evecs_aligned, device="cuda:0"
        )
        heat_diff_before_after.append(combine_images(heat_diff_before, heat_diff_after))

    print(
        "\n\nCopy and paste the followings to see aligned evecs when changing the",
        f"ALBO angle to {all_angles}",
    )
    print("for img in set_of_images:\n    show_image(img)")

    print(
        "\n\nCopy and paste the followings to see heat diff before and after evecs",
        f"alignement when changing the ALBO angle to {all_angles}",
    )
    print("for img in heat_diff_before_after:\n    show_image(img)")
