import os
from dataclasses import asdict
from pathlib import Path

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

import torch

from IPython import get_ipython
import trimesh
import argparse
import heatsplats
from heatsplats.data import MeshSamplerDataModule
from heatsplats.modules import HeatKernelDensityKNN
from heatsplats.rendering.heat_kernels_density_renderer_knn import (
    HeatKernelsDensityRendererKNN,
)
from heatsplats.rendering.heat_kernels_renderer_knn import HeatKernelsRendererKNN
from heatsplats.trainers import BaseTrainer
from heatsplats.utils import (
    big_trimesh_pcl,
    compute_all_image_metrics,
    load_config,
    mibitmaps2torch,
    repr_patches,
    seed_everything,
    config_to_primitive,
)
from heatsplats.utils.video import combine_videos, save_video, show_video

__all__ = ["repr_patches", "show_video"]


try:
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
        print("Running in a Jupyter Notebook. Enabling autoreload...")
        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")
except NameError:
    pass


def infer_default_config_path(ckpt_path: str) -> str | None:
    ckpt = Path(ckpt_path).resolve()
    parsed_cfg = ckpt.parent.parent / "configs" / "parsed.yaml"
    if parsed_cfg.exists():
        return str(parsed_cfg)
    return None


def load_datamodule_and_trainer_only(args, extras):
    cfg = load_config(args.config, cli_args=extras, n_gpus=1)
    seed_everything(cfg.seed)

    datamodule: MeshSamplerDataModule = heatsplats.find(cfg.data_type)(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")

    cfg.trainer.density_controllers = []
    trainer: BaseTrainer = heatsplats.find(cfg.trainer_type)(
        cfg.trainer, datamodule, renderer_cfg=cfg.renderer
    )
    return cfg, datamodule, trainer


def render_rotating(renderer, mi_mesh, n_rotating_frames: int):
    if n_rotating_frames == 1:
        img = renderer.render(mi_mesh, denoise=True)
        return mi.Bitmap(img).convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True,
        )
    return renderer.rotating_video(mi_mesh, n_rotating_frames)


def render_prediction(trainer: BaseTrainer, n_rotating_frames: int):
    renderer = HeatKernelsRendererKNN(trainer.cfg.renderer)
    renderer.mega_kernel(
        trainer.cfg.renderer_mega_kernel, no_loops=True, no_opt_calls=True
    )

    mi_mesh = renderer.mesh_to_mitsuba(
        trainer.datamodule.mesh, trainer.mesh, trainer.model, trainer.eigalbo_interp
    )
    trainer.prepare_knn(False)
    try:
        return render_rotating(renderer, mi_mesh, n_rotating_frames)
    finally:
        renderer.flush_cache()
        trainer.reset_knn()


def build_density_model(trainer: BaseTrainer, ckpt_path: str, density_mode: str):
    density_cfg = config_to_primitive(trainer.model.cfg)
    print(density_cfg)
    density_cfg["density_mode"] = density_mode
    density_model = HeatKernelDensityKNN(density_cfg, trainer.mesh)
    density_model.load_torch(ckpt_path)
    density_model.eval()
    return density_model


def ensure_video_frames(rendering):
    if isinstance(rendering, list):
        return rendering
    return [rendering]


def render_density(
    trainer: BaseTrainer,
    density_model: HeatKernelDensityKNN,
    n_rotating_frames: int,
):
    renderer = HeatKernelsDensityRendererKNN(trainer.cfg.renderer)
    renderer.mega_kernel(
        trainer.cfg.renderer_mega_kernel, no_loops=True, no_opt_calls=True
    )

    mi_mesh = renderer.mesh_to_mitsuba(
        trainer.datamodule.mesh,
        trainer.mesh,
        density_model,
        trainer.eigalbo_interp,
    )
    density_model.prepare_kernels(trainer.mesh, trainer.eigalbo_interp, False)
    try:
        return render_rotating(renderer, mi_mesh, n_rotating_frames)
    finally:
        renderer.flush_cache()
        density_model.reset(trainer.eigalbo_interp)


if __name__ == "__main__":
    os.environ["CUDA_HOME"] = "/vol/cuda/12.9.0/"
    args_dict = {
        # "config": "configs/uv_texture_fitting_knn.yaml",
        "config": None,
        "gpu": "0",
        "verbose": True,
    }
    extras_dict = {
        "data.mesh_path": "/data2/objaverse/hf-objaverse-v1/glbs/000-000/0413d92d70f24a68b3afe8643301c515.glb",
        "trainer.eigen_albo.mesh_path": "/data2/objaverse/hf-objaverse-v1/glbs/000-000/0413d92d70f24a68b3afe8643301c515.glb",
        "trainer.eigen_albo.error_if_not_precomputed": True,
        "optim.iters": 0,
        "optim.save_model": False,
        "optim.save_logs": False,
        "trainer.density_controllers": [],
        "renderer.point_batching": 512,
        "renderer.n_rotating_frames": 5,
        "renderer.integrator_config.type": "prb",
        "renderer.integrator_config.meta.max_depth": 2,
        "trainer.renderer_mega_kernel": False,
        "renderer.camera_config.tile_size_heatkernels": None,
        # "trainer.model.range_enforcement_type": "pgd",
        # "trainer.model.init_kernel_edge_type": "uniform",
        # "trainer.eigen_albo.use_euclidian_distance": False,
        # "trainer.eigen_albo.faiss.index_type": "flat",
        "ckpt_path": "/data/home/ck223/heatsplats/outputs/ablations/uv-knn-fix/benchmark_run/benchmark_trainable_825ac_00048_48_filename=hf-objaverse-v1_glbs_000-000_0413d92d70f24a68b3afe8643301c515_glb_2026-03-11_13-26-40/output/hf-objaverse-v1_glbs_000-000_0413d92d70f24a68b3afe8643301c515/ckpts/model.pt",
        "output_path": None,
    }

    args = argparse.Namespace(**args_dict)

    ckpt_path = extras_dict.pop("ckpt_path")
    output_path = extras_dict.pop("output_path")

    if args.config is None:
        args.config = infer_default_config_path(ckpt_path)
        if args.config is None:
            raise ValueError(
                "No config provided and no parsed.yaml found next to the checkpoint."
            )

    extras = [f"{k}={v}" for k, v in extras_dict.items()]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg, datamodule, trainer = load_datamodule_and_trainer_only(args, extras)
    trainer.model.load_torch(ckpt_path)
    trainer.model.eval()

    n_frames = cfg.renderer.n_rotating_frames
    optimisation: BaseTrainer = trainer
    mesh: trimesh.Trimesh = datamodule.mesh

    gt_rend = trainer.render_gt(n_frames)
    pred_rend = render_prediction(trainer, n_frames)

    coverage_model = build_density_model(trainer, ckpt_path, "coverage")
    coverage_rend = render_density(trainer, coverage_model, n_frames)

    centers_model = build_density_model(trainer, ckpt_path, "centers")
    centers_model.cfg.density_min = 0.0
    centers_model.cfg.density_max = 10.0
    centers_model.cfg.center_sigma = 0.001
    centers_model.cfg.log_beta = 0.0
    centers_rend = render_density(trainer, centers_model, n_frames)

    gt_frames = ensure_video_frames(gt_rend)
    pred_frames = ensure_video_frames(pred_rend)
    coverage_frames = ensure_video_frames(coverage_rend)
    centers_frames = ensure_video_frames(centers_rend)

    combined_rend = combine_videos(
        gt_frames,
        pred_frames,
        coverage_frames,
        centers_frames,
    )

    gt = mibitmaps2torch(gt_frames)
    pred = mibitmaps2torch(pred_frames)
    metrics = compute_all_image_metrics(pred, gt)

    if output_path is None:
        output_path = str(Path(ckpt_path).resolve().parent / "gt_pred_cov_centers.mp4")
    save_video(combined_rend, output_path)

    v_scene = trimesh.Scene([mesh, big_trimesh_pcl(optimisation.kernel_centres)])

    print("Config:", args.config)
    print("Checkpoint:", ckpt_path)
    print("Panel order: GT | Prediction | Coverage density | Centers density")
    print("Metrics:", metrics)
    print("Saved video:", output_path)
    print("You can now visualise the followings:")
    print("  - Mesh with kernel centres: v_scene.show()")
    print("  - GT renderings: show_video(gt_rend)")
    print("  - Prediction renderings: show_video(pred_rend)")
    print("  - Coverage density renderings: show_video(coverage_rend)")
    print("  - Centers density renderings: show_video(centers_rend)")
    print("  - Combined renderings: show_video(combined_rend)")
