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

from heatsplats.modules import Mesh
from heatsplats.utils import (
    load_mesh,
    get_anisotropic_lbo,
    compute_mesh_laplacian,
    compute_eig_laplacian,
    heat_diffusion,
    soft_step,
    rescaled_soft_step,
    combine_images,
    show_image,
)
from heatsplats.rendering.vertex_colours_renderer import VertexColoursRenderer


def heat_diffuse_single_kernel(
    mesh: Mesh,
    angle: torch.Tensor,
    scale: torch.Tensor,
    diff_times: torch.Tensor,
    k_eig: int = 256,
    idxs: torch.Tensor = None,
    device: str = "cuda:0",
    i: float = 1.0,
    local_direction=None,
):

    l_evals, l_evecs = [], []
    mass = None
    for a, s in zip(angle, scale):
        # lapl, mass = get_anisotropic_lbo_old(
        #     mesh.verts,
        #     mesh.faces.T,
        #     mesh.fnorms,
        #     rotation_angle=a.cpu().detach().numpy(),
        #     anisotropy=s.cpu().detach().numpy(),
        # )
        lapl, mass = get_anisotropic_lbo(
            mesh.verts,
            mesh.faces.T,
            mesh.fnorms,
            rotation_angle=a.cpu().detach().numpy(),
            anisotropy=s.cpu().detach().numpy(),
            local_direction=local_direction,
        )
        # print(
        #     "TODO: on a plane the principal curvatures are always zero so the laplacian is zero, fix this"
        # )

        # lapl, mass = compute_mesh_laplacian(
        #     mesh.verts.cpu().detach().numpy(), mesh.faces.cpu().detach().numpy()
        # )

        lapl = lapl.astype(np.float32)
        mass = mass.astype(np.float32)

        evals, evecs = compute_eig_laplacian(lapl, mass, k_eig)
        l_evals.append(torch.tensor(evals))
        l_evecs.append(torch.tensor(evecs))
    evals = torch.stack(l_evals, dim=0).to(device)
    # evals[:, i] = evals[:, i] * 100
    evecs = torch.stack(l_evecs, dim=0).to(device)
    # evecs[:, :, i] = evecs[:, :, i] * 10
    mass = torch.tensor(mass).to(device)

    if idxs is None:
        idxs = torch.randint(0, mesh.N_verts, (3,), device=device)

    B, V = scale.shape[0], mesh.N_verts
    v_colours = torch.zeros([B, V, 1], device=device)
    v_colours[torch.arange(B), idxs, 0] = 1.0

    # v_colours = heat_diffusion(v_colours, mass, evals, evecs, diff_times)
    ####################################################################################
    basisT = evecs.transpose(-2, -1)
    x_spec = torch.matmul(basisT, v_colours * mass.unsqueeze(-1))
    diffusion_coefs = torch.exp(-evals * diff_times.unsqueeze(-1)).unsqueeze(-1)
    # diffusion_coefs[:, i] = diffusion_coefs[:, i] * 100.0
    x_diffuse_spec = diffusion_coefs * x_spec
    v_colours = torch.matmul(evecs, x_diffuse_spec)
    ####################################################################################
    v_colours = v_colours / (v_colours[torch.arange(B), idxs, :].unsqueeze(1) + 1e-8)

    rand_colours = torch.rand((B, 3), device=device)
    v_colours = v_colours * rand_colours.unsqueeze(1)
    v_colours = v_colours.sum(dim=0)

    return v_colours


if __name__ == "__main__":

    # fname = "../objects/spot/spot_triangulated.obj"
    # fname = "../objects/square_mesh.ply"
    # tri_mesh = load_mesh(fname, merge_tex=False, bake_vert_colors=False)
    tri_mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)

    our_mesh = Mesh.from_trimesh(tri_mesh, device="cuda:0")
    vc_renderer = VertexColoursRenderer({"camera_config": {"azimuth_deg": 0}})
    # vc_renderer = VertexColoursRenderer({"camera_config": {"azimuth_deg": -90}})

    images_vertices = []
    # for i in range(0, 20):
    # for i in [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
    # for i in [0.5, 1, 2.5, 5, 7.5, 10]:
    for i in np.linspace(0, 180, 9)[:-1]:
        angle = torch.deg2rad(torch.tensor([i, i], device="cuda:0"))
        scale = torch.tensor([10, 10], device="cuda:0")
        diff_times = torch.tensor([1e-3, 1e-3], device="cuda:0")

        # idxs = torch.tensor([182, 170], device="cuda:0")  # for square
        idxs = torch.tensor([2265, 61], device="cuda:0")  # for icosphere
        # idxs = torch.randint(0, our_mesh.N_verts, (2,), device="cuda:0")
        # print(f"i: {i}, idxs: {idxs.cpu().detach().numpy()}")

        v_colours = heat_diffuse_single_kernel(
            our_mesh,
            angle,
            scale,
            diff_times,
            idxs=idxs,
            device="cuda:0",
            i=i,
            local_direction=None,
        )

        out_mesh = tri_mesh.copy()
        out_mesh.visual = trimesh.visual.ColorVisuals(
            out_mesh, vertex_colors=v_colours.cpu().detach().numpy()
        )
        mi_mesh = vc_renderer.mesh_to_mitsuba(out_mesh)
        images_vertices.append(vc_renderer.render(mi_mesh, denoise=True))

    combined_image = combine_images(*images_vertices)
    print(f"show all angles with: show_image(combined_image)")
