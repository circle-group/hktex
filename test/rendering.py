import sys
from pathlib import Path
import matplotlib.pyplot as plt
import mitsuba as mi

sys.path.append(str(Path(__file__).resolve().parent.parent))

from heatsplats.utils import load_mesh
from heatsplats.rendering.vertex_colours_renderer import VertexColoursRenderer
from heatsplats.rendering.uv_texture_renderer import UVTextureRenderer

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
