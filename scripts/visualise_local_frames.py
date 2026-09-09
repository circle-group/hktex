import sys
from pathlib import Path
import os

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import torch
from hktex.utils import compute_aligned_frame

if __name__ == "__main__":
    import igl
    import trimesh
    import numpy as np

    import meshplot as mp
    from hktex import utils

    fname = "../objects/spot/spot_triangulated.obj"
    mesh = utils.load_mesh(fname)
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    v, f = mesh.vertices, mesh.faces
    avg = igl.avg_edge_length(v, f) / 2.0

    # Local reference frames based on Principla curvature ##############################
    v1, v2, k1, k2 = igl.principal_curvature(v, f)
    h2 = 0.5 * (k1 + k2)
    myplot = mp.plot(v, f, shading={"wireframe": False}, return_plot=True)

    myplot.add_lines(v + v1 * avg, v - v1 * avg, shading={"line_color": "red"})
    myplot.add_lines(v + v2 * avg, v - v2 * avg, shading={"line_color": "green"})

    # Local reference frames aligned with axes and smoothed ############################
    v3, v4, _ = compute_aligned_frame(
        verts=torch.tensor(v),
        faces=torch.tensor(f),
        normals=torch.tensor(mesh.vertex_normals),
        num_iterations=5,
    )
    v3, v4 = v3.cpu().detach().numpy(), v4.cpu().detach().numpy()

    myplot2 = mp.plot(v, f, shading={"wireframe": False}, return_plot=True)
    myplot2.add_lines(v + v3 * avg, v - v3 * avg, shading={"line_color": "red"})
    myplot2.add_lines(v + v4 * avg, v - v4 * avg, shading={"line_color": "green"})
