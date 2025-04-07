import math
import trimesh
import torch

import numpy as np

from heatsplats.utils import (
    load_mesh,
    get_anisotropic_lbo,
    compute_eig_laplacian,
    heat_diffusion,
)

if __name__ == "__main__":
    # mesh = load_mesh("objects/mech_drone.glb", show=False)
    mesh = load_mesh("../objects/spot/spot_triangulated.obj", show=False)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    fnorm = np.array(mesh.face_normals)
    # a, b = compute_mesh_laplacian(verts, faces)
    # c, d = get_mesh_laplacian(torch.tensor(verts), torch.tensor(faces).T)

    a = 100
    r = math.radians(0)
    print(a)
    # lapl, massvec = compute_mesh_laplacian(verts, faces)

    lapl, mass = get_anisotropic_lbo(
        torch.tensor(verts),
        torch.tensor(faces).T,
        torch.tensor(fnorm),
        rotation_angle=r,
        anisotropy=a,
    )

    eval, evecs = compute_eig_laplacian(lapl=lapl, massvec=mass, k_eig=256)

    eval = torch.tensor(eval).unsqueeze(0).contiguous()
    evecs = torch.tensor(evecs).unsqueeze(0).contiguous()
    mass = torch.tensor(mass).unsqueeze(0).contiguous()

    colours = torch.ones_like(torch.tensor(mesh.vertices))
    colours = colours.unsqueeze(0)

    # c[10, :] = np.array([255, 0, 0])

    # for _ in range(1000):
    c_i = torch.zeros_like(colours)
    # c_i[:, torch.randint(c_i.shape[1] - 1, (1,)).item(), :] = torch.rand(
    #     3
    # ).unsqueeze(0)
    # c_i[torch.randint(c_i.shape[0] - 1, (1,)).item(), :] = torch.tensor(
    #     [1.0, 0, 0]
    # ).unsqueeze(0)
    c_i[:, 0, :] = torch.tensor([1.0, 0, 0]).unsqueeze(0)
    c_i = heat_diffusion(c_i, mass, eval, evecs, torch.tensor(0.1))
    colours += c_i

    # colours = colours * -1 + 1
    colours = (colours - colours.min()) / (colours.max() - colours.min())
    colours *= 255
    colours = colours.squeeze().numpy()

    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colours)
