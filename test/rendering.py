import sys
from pathlib import Path
import matplotlib.pyplot as plt
import mitsuba as mi

sys.path.append(str(Path(__file__).resolve().parent.parent))

from heatsplats.utils import load_mesh
from heatsplats.rendering.vertex_colours_renderer import VertexColoursRenderer
from heatsplats.rendering.uv_texture_renderer import UVTextureRenderer
from heatsplats.rendering.heat_kernels_renderer import HeatKernelsRenderer
from heatsplats.modules import Mesh, Model, EigenAlboInterpolation

if __name__ == "__main__":
    fname = "../objects/spot/spot_triangulated.obj"

    # Test UVTextureRenderer
    mesh = load_mesh(fname, merge_tex=False, bake_vert_colors=False)
    uv_texture_renderer = UVTextureRenderer(dict())
    mi_mesh = uv_texture_renderer.mesh_to_mitsuba(mesh)
    image = uv_texture_renderer.render(mi_mesh)
    bitmap = mi.Bitmap(image).convert(srgb_gamma=True)

    # Test VertexColoursRenderer
    mesh = load_mesh(fname, merge_tex=False, bake_vert_colors=True)
    vertex_colours_renderer = VertexColoursRenderer(dict())
    mi_mesh = vertex_colours_renderer.mesh_to_mitsuba(mesh)
    image = vertex_colours_renderer.render(mi_mesh)
    bitmap2 = mi.Bitmap(image).convert(srgb_gamma=True)

    # Test HeatKernelsRenderer
    fname = "../objects/bob/bob_tri.obj"
    tri_mesh = load_mesh(fname, merge_tex=True, bake_vert_colors=False)
    our_mesh = Mesh.from_trimesh(tri_mesh, device="cuda:0")
    model_cfg = {
        "weights": None,
        "n_sources": 400,
        "out_dim": 3,
        "kernel_dim": 32,
        "out_net": True,
        "normalize_colours": False,
    }
    eigalbo_config = {
        "k_eig": 256,
        "use_precomputed": True,
        "precompute_anisotropies": [1, 2.5, 5, 7.5, 10, 25, 50, 75, 100],
        "precompute_angles_every_deg": 30,
        "mesh_path": "../objects/bob/bob_tri.obj",
        "precomputed_name": "eigen_albo",
    }
    model = Model(model_cfg, our_mesh)
    model.load_torch("outputs/hk_bob.pt")
    eigalbo_interp = EigenAlboInterpolation(eigalbo_config, our_mesh)
    hk_renderer = HeatKernelsRenderer(dict())
    hk_renderer.mega_kernel(False)
    mi_mesh = hk_renderer.mesh_to_mitsuba(tri_mesh, our_mesh, model, eigalbo_interp)
    image = hk_renderer.render(mi_mesh)
    hk_renderer.flush_cache()
    bitmap3 = mi.Bitmap(image).convert(srgb_gamma=True)
