import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
import drjit as dr
import mitsuba as mi
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hktex.utils import load_mesh, load_config, ExperimentConfig, show_video
from hktex.rendering.vertex_colours_renderer import VertexColoursRenderer
from hktex.rendering.uv_texture_renderer import UVTextureRenderer
from hktex.rendering.heat_kernels_renderer import HeatKernelsRenderer
from hktex.modules import Mesh, HeatKernelTexture, EigenAlboInterpolation

if __name__ == "__main__":
    fname = "../objects/spot/spot_triangulated.obj"

    # Test UVTextureRenderer
    mesh = load_mesh(fname, merge_tex=False, bake_vert_colors=False)
    uv_texture_renderer = UVTextureRenderer(dict())
    mi_mesh = uv_texture_renderer.mesh_to_mitsuba(mesh)
    image = uv_texture_renderer.render(mi_mesh)
    bitmap = mi.Bitmap(image).convert(srgb_gamma=True)
    frames = uv_texture_renderer.rotating_video(mi_mesh, n_frames=20)

    # Test VertexColoursRenderer
    mesh = load_mesh(fname, merge_tex=False, bake_vert_colors=True)
    vertex_colours_renderer = VertexColoursRenderer(dict())
    mi_mesh = vertex_colours_renderer.mesh_to_mitsuba(mesh)
    image = vertex_colours_renderer.render(mi_mesh)
    bitmap2 = mi.Bitmap(image).convert(srgb_gamma=True)

    # Test HeatKernelsRenderer
    # experiment = None
    experiment = "outputs/uv-texture-fitting/spot_triangulated@20250429-180141"
    if experiment is None:
        fname = "../objects/bob/bob_tri.obj"
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
        ckpt_name = "outputs/hk_bob.pt"
    else:
        cfg_path = os.path.join(experiment, "configs/parsed.yaml")
        main_cfg: ExperimentConfig = load_config(cfg_path)
        fname = main_cfg.data["mesh_path"]
        model_cfg = main_cfg.trainer["model"]
        eigalbo_config = main_cfg.trainer["eigen_albo"]
        ckpt_name = os.path.join(
            main_cfg.trial_dir, "ckpts", main_cfg.optim.save_model_name
        )
    # dr.set_log_level(dr.LogLevel.Info)
    tri_mesh = load_mesh(fname, merge_tex=True, bake_vert_colors=False)
    our_mesh = Mesh.from_trimesh(tri_mesh, device="cuda:0")
    model = HeatKernelTexture(model_cfg, our_mesh)
    model.load_torch(ckpt_name)
    eigalbo_interp = EigenAlboInterpolation(eigalbo_config, our_mesh)
    hk_renderer = HeatKernelsRenderer(dict())
    hk_renderer.mega_kernel(False)
    t0 = time.time()
    mi_mesh = hk_renderer.mesh_to_mitsuba(tri_mesh, our_mesh, model, eigalbo_interp)
    image = hk_renderer.render(mi_mesh, denoise=True)
    t1 = time.time()
    print(f"Rendering time: {t1 - t0:.2f} seconds")
    hk_renderer.flush_cache()
    bitmap3 = mi.Bitmap(image).convert(srgb_gamma=True)
