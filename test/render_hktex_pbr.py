import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
import drjit as dr
import mitsuba as mi
import time
import glob
import torch
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from heatsplats.utils import (
    load_mesh,
    load_config,
    ExperimentConfig,
    show_video,
    save_video,
)
from heatsplats.modules import Mesh, HeatKernelTexture, EigenAlboInterpolation
import heatsplats


def render(
    experiment: str,
    ckpt_name: str,
    n_frames: int = 1,
    fps: int = 30,
    output_dir: str | None = None,
    mesh_dir: str | None = None,
    save_format: str = "both",
    random_kernels: int | None = None,
    random_threshold_scale: float = 0.01,
):
    cfg_path = os.path.join(experiment, "configs/parsed.yaml")
    main_cfg: ExperimentConfig = load_config(cfg_path)
    fname = main_cfg.data["mesh_path"]

    if mesh_dir is not None:
        target = "hf-objaverse-v1"
        if target in fname:
            idx = fname.find(target)
            if target in mesh_dir:
                after_target = fname[idx + len(target) :].lstrip("/")
                fname = os.path.join(mesh_dir, after_target)
            else:
                from_target = fname[idx:]
                fname = os.path.join(mesh_dir, from_target)
        else:
            fname = os.path.join(mesh_dir, os.path.basename(fname))

    # Handle coeus cluster path or machine-specific prefixes (/data/sf3018, coeus)
    import re
    if not os.path.exists(fname):
        if "coeus" in fname:
            fname_clean = re.sub(r".*coeus[^/]*/", "", fname)
            cand = os.path.join(mesh_dir or "/data2/objaverse", fname_clean)
            if os.path.exists(cand):
                fname = cand
        if fname.startswith("/data/sf3018/"):
            cand = fname.replace("/data/sf3018/", "/data2/")
            if os.path.exists(cand):
                fname = cand

    if not os.path.exists(fname):
        cand1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", fname))
        cand2 = os.path.join("/homes/sf3018/Documents/objects", os.path.basename(fname))
        if os.path.exists(cand1):
            fname = cand1
        elif os.path.exists(cand2):
            fname = cand2
        elif mesh_dir and os.path.exists(mesh_dir):
            matches = glob.glob(
                os.path.join(mesh_dir, f"**/{os.path.basename(fname)}"),
                recursive=True,
            )
            if matches:
                fname = matches[0]

    print(f"Using mesh path: {fname}")
    main_cfg.data["mesh_path"] = fname

    trainer_cfg = main_cfg.trainer
    if "model" in trainer_cfg:
        raw_model_cfg = trainer_cfg["model"]
    elif "network" in trainer_cfg and "model" in trainer_cfg["network"]:
        raw_model_cfg = trainer_cfg["network"]["model"]
    else:
        raise KeyError("Could not find 'model' in trainer configuration")

    if "eigen_albo" in trainer_cfg:
        eigalbo_config = trainer_cfg["eigen_albo"]
    elif "network" in trainer_cfg and "eigen_albo" in trainer_cfg["network"]:
        eigalbo_config = trainer_cfg["network"]["eigen_albo"]
    else:
        eigalbo_config = {}

    model_cfg = OmegaConf.to_container(raw_model_cfg, resolve=True)
    eigalbo_config["mesh_path"] = fname

    random_suffix = f"_random_{random_kernels}" if random_kernels is not None else ""

    if random_kernels is not None:
        model_cfg["n_sources"] = random_kernels
        print(
            f"Option selected: Rendering with {random_kernels} randomly ",
            f"initialized kernels (suffix: '{random_suffix}').",
        )
    else:
        print("Option selected: Rendering trained kernels from checkpoint.")

    tri_mesh = load_mesh(fname, merge_tex=True, bake_vert_colors=False)
    our_mesh = Mesh.from_trimesh(tri_mesh, device="cuda:0")

    # Check if the environment map exists
    user_envmap = "/homes/sf3018/Documents/objects/venice_sunset_1k.exr"
    if os.path.exists(user_envmap):
        envmap_abs_path = user_envmap
        print(f"Using environment map: {envmap_abs_path}")
    else:
        raise FileNotFoundError(f"Environment map not found: {user_envmap}")

    renderer_config = {
        "emitter_config": {
            "envmap_path": envmap_abs_path,
            "envmap_scale": 1.0,
        },
        "integrator_config": {
            "type": "path",
            "hide_emitters": True,  # invisible background
        },
        "ground_plane_config": {
            "activated": False,
        },
        "camera_config": {
            "img_width": 1200,
            "img_height": 1200,
            "camera_distance": 5,
            "azimuth_deg": -40,  # 145: pilot, 40: rex and ding
            "elevation_deg": 20,
        },
    }

    use_knn = (
        main_cfg.trainer.get("use_knn_implementation", False)
        or "knn" in main_cfg.trainer_type.lower()
    )

    if use_knn:
        print("Using KNN implementation based on experiment config.")
        from heatsplats.modules.heat_kernel_texture_knn import HeatKernelTextureKNN
        from heatsplats.modules.eigen_albo_knn import EigenAlboInterpolationKNN
        from heatsplats.rendering.heat_kernels_renderer_knn import (
            HeatKernelsRendererKNN,
        )

        model = HeatKernelTextureKNN(model_cfg, our_mesh)
        if random_kernels is None:
            if ckpt_name.endswith(".npz"):
                model.load_numpy_npz(ckpt_name)
            else:
                model.load_torch(ckpt_name)
        eigalbo_interp = EigenAlboInterpolationKNN(eigalbo_config, our_mesh)
        hk_renderer = HeatKernelsRendererKNN(renderer_config)
    else:
        print("Using standard implementation based on experiment config.")
        from heatsplats.rendering.heat_kernels_renderer import HeatKernelsRenderer

        model = HeatKernelTexture(model_cfg, our_mesh)
        if random_kernels is None:
            if ckpt_name.endswith(".npz"):
                model.load_numpy_npz(ckpt_name)
            else:
                model.load_torch(ckpt_name)
        eigalbo_interp = EigenAlboInterpolation(eigalbo_config, our_mesh)
        hk_renderer = HeatKernelsRenderer(renderer_config)

    # Scale thresholds when randomly initialized
    if random_kernels is not None:
        class_type = type(model)

        def get_scaled_thresholds(self):
            return self._thresholds_act(self._thresholds) * random_threshold_scale

        class_type.thresholds = property(get_scaled_thresholds)
        print(f"Scaled random kernels' thresholds by factor: {random_threshold_scale}")

        # Force base color to be black when randomly initialized
        assert model.out_net is None
        with torch.no_grad():
            model._mean_colour.fill_(0.0)

    materials = {
        "albedo": {},
        "gold_metallic": {
            "roughness": 0.15,
            "metallic": 1.0,
            "spec_tint": 0.5,
        },
        "shiny_plastic": {
            "roughness": 0.1,
            "metallic": 0.0,
            "specular": 0.5,
        },
        "matte_chalk": {
            "roughness": 0.8,
            "metallic": 0.0,
        },
        "brushed_anisotropic": {
            "roughness": 0.25,
            "metallic": 0.9,
            "anisotropic": 0.8,
        },
        "clearcoat_gloss": {
            "roughness": 0.4,
            "metallic": 0.7,
            "clearcoat": 1.0,
            "clearcoat_gloss": 0.8,
        },
    }

    hk_renderer.mega_kernel(False)

    if use_knn:  # Build the FAISS index for manual rendering if using KNN
        model.prepare_kernels(our_mesh, eigalbo_interp, save_barycentric=False)

    if output_dir is None:
        output_dir = os.path.join(experiment, "pbr_renders")

    os.makedirs(output_dir, exist_ok=True)

    is_video = n_frames > 1

    if not is_video:
        print("Rendering ground plane pass (reusable)...")
        hk_renderer.cfg.ground_plane_config.activated = True
        hk_renderer.reset_scene()

        scene_dict = hk_renderer.configure_scene()
        scene_plane = mi.load_dict(scene_dict)
        I_plane = hk_renderer.render(scene=scene_plane, denoise=True)
        t_plane = mi.TensorXf(I_plane).torch()
    else:
        t_plane = None

    for name, params in materials.items():
        print(f"Rendering material '{name}' with params: {params}")
        t0 = time.time()

        mi_mesh = hk_renderer.mesh_to_mitsuba(
            tri_mesh, our_mesh, model, eigalbo_interp, bsdf_additional_settings=params
        )

        if is_video:
            frames = hk_renderer.rotating_video(
                mi_mesh=mi_mesh, n_frames=n_frames, shadow_catcher=True
            )
            t1 = time.time()
            print(f"  - Render video time: {t1 - t0:.4f}s")
            if save_format in ("video", "both"):
                out_name = f"{name}{random_suffix}.mp4"
                out_path = os.path.join(output_dir, out_name)
                save_video(frames, out_path, fps=fps)
                print(f"  - Saved video to: {out_path}")

            if save_format in ("frames", "both"):
                frames_dir = os.path.join(output_dir, f"{name}{random_suffix}_frames")
                os.makedirs(frames_dir, exist_ok=True)
                for idx, frame_bmp in enumerate(frames):
                    frame_bmp.write(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
                print(f"  - Saved {len(frames)} transparent frames to: {frames_dir}")
        else:
            I_shadow = hk_renderer.render_shadow_catcher(
                mi_mesh=mi_mesh, t_plane=t_plane
            )
            t1 = time.time()
            print(f"  - Render & composite time: {t1 - t0:.4f}s")

            # Convert and write to file
            bitmap = mi.Bitmap(I_shadow).convert(
                pixel_format=mi.Bitmap.PixelFormat.RGBA,
                component_format=mi.Struct.Type.UInt8,
                srgb_gamma=True,
            )
            out_name = f"{name}{random_suffix}.png"
            out_path = os.path.join(output_dir, out_name)
            bitmap.write(out_path)
            print(f"  - Saved to: {out_path}")

    # Render albedo with constant emitter (default lighting: envmap_path = None, radiance = 1.0)
    print("Rendering albedo with constant emitter...")
    hk_renderer.cfg.emitter_config.envmap_path = None
    hk_renderer.cfg.emitter_config.radiance = 1.0
    hk_renderer.reset_scene()

    mi_mesh = hk_renderer.mesh_to_mitsuba(
        tri_mesh, our_mesh, model, eigalbo_interp, bsdf_additional_settings={}
    )

    if is_video:
        frames = hk_renderer.rotating_video(
            mi_mesh=mi_mesh, n_frames=n_frames, shadow_catcher=True
        )
        if save_format in ("video", "both"):
            out_name = f"albedo_constant{random_suffix}.mp4"
            out_path = os.path.join(output_dir, out_name)
            save_video(frames, out_path, fps=fps)
            print(f"  - Saved video to: {out_path}")

        if save_format in ("frames", "both"):
            frames_dir = os.path.join(
                output_dir, f"albedo_constant{random_suffix}_frames"
            )
            os.makedirs(frames_dir, exist_ok=True)
            for idx, frame_bmp in enumerate(frames):
                frame_bmp.write(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
            print(f"  - Saved {len(frames)} transparent frames to: {frames_dir}")
    else:
        I_shadow_constant = hk_renderer.render_shadow_catcher(
            mi_mesh=mi_mesh, t_plane=None
        )

        bitmap = mi.Bitmap(I_shadow_constant).convert(
            pixel_format=mi.Bitmap.PixelFormat.RGBA,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True,
        )
        out_name = f"albedo_constant{random_suffix}.png"
        out_path = os.path.join(output_dir, out_name)
        bitmap.write(out_path)
        print(f"  - Saved to: {out_path}")

    # Reset FAISS index if using KNN
    if use_knn:
        model.reset(eigalbo_interp)

    hk_renderer.flush_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render PBR materials and shadow catching for HeatKernelTexture models."
    )
    parser.add_argument(
        "--txt_path",
        "-f",
        type=str,
        default=None,
        help="Path to txt file containing list of partial match filenames.",
    )
    parser.add_argument(
        "--root_dir",
        "-r",
        type=str,
        default=None,
        help="Folder root containing all experiment runs.",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default=None,
        help="Folder where to save all experiment runs.",
    )
    parser.add_argument(
        "--glob_pattern",
        "-g",
        type=str,
        default="outputs/uv-texture-fitting/rex*@*",
        help="Glob pattern to search for available experiment runs.",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=1,
        help="Number of frames for rendering. If n_frames > 1, renders a rotating video.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output video.",
    )
    parser.add_argument(
        "--mesh_dir",
        "-m",
        type=str,
        default=None,
        help="Root or prefix directory for mesh files (replaces path before 'hf-objaverse-v1').",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["video", "frames", "both"],
        default="both",
        help="Format to save when n_frames > 1: 'video' for .mp4, 'frames' for transparent PNGs, or 'both'.",
    )
    parser.add_argument(
        "--random_kernels",
        type=int,
        default=None,
        help="Set number of randomly initialized kernels to render instead of loading checkpoint.",
    )
    parser.add_argument(
        "--random_threshold_scale",
        type=float,
        default=0.01,
        help="Scale factor for thresholds when rendering random kernels (default: 0.01).",
    )
    args = parser.parse_args()

    if args.txt_path and args.root_dir:
        # Read txt file of partial match names and match folders in root_dir
        with open(args.txt_path, "r") as f:
            names = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

        missing_folders = []
        missing_ckpts = []

        for name in names:
            search_roots = [args.root_dir] if args.root_dir else []
            for fallback in [
                "/data/home/ck223/heatsplats/outputs/uv-mitsuba-fitting",
                "/data2/home/sf3018/hktex/multiview/hs-ray",
                "/data2/home/sf3018/hktex/multiview/hs-ray-2",
                "/data2/home/sf3018/hktex/benchmark_hktex/benchmark_run",
                "/data/home/ck223/heatsplats/outputs",
            ]:
                if os.path.exists(fallback) and fallback not in search_roots:
                    search_roots.append(fallback)

            matched_run = None
            query_patterns = [name]
            if "_" in name:
                query_patterns.append(name.split("_")[-1])

            for sroot in search_roots:
                for q in query_patterns:
                    matches = glob.glob(os.path.join(sroot, f"*{q}*"))
                    if not matches:
                        matches = glob.glob(os.path.join(sroot, f"**/*{q}*"), recursive=True)
                    if matches:
                        # Prefer folder that actually contains configs/parsed.yaml or ckpts
                        for m in matches:
                            if os.path.exists(os.path.join(m, "configs/parsed.yaml")) or glob.glob(os.path.join(m, "**/parsed.yaml"), recursive=True):
                                matched_run = m
                                break
                        if not matched_run:
                            matched_run = matches[0]
                        break
                if matched_run:
                    break

            if matched_run:
                output_subdirs = glob.glob(os.path.join(matched_run, "output/*"))
                experiment = output_subdirs[0] if output_subdirs else matched_run

                pts = glob.glob(os.path.join(experiment, "ckpts/*.npz")) + glob.glob(
                    os.path.join(experiment, "ckpts/*.pt")
                )
                if not pts:
                    pts = glob.glob(
                        os.path.join(matched_run, "**/ckpts/*.npz"), recursive=True
                    ) + glob.glob(
                        os.path.join(matched_run, "**/ckpts/*.pt"), recursive=True
                    )

                if pts:
                    ckpt_name = pts[0]
                    print(f"Using experiment run: {experiment}")
                    print(f"Using checkpoint: {ckpt_name}")
                    if args.out_dir is None:
                        output_dir = None
                    else:
                        output_dir = os.path.join(args.out_dir, name)
                    render(
                        experiment,
                        ckpt_name,
                        n_frames=args.n_frames,
                        fps=args.fps,
                        output_dir=output_dir,
                        mesh_dir=args.mesh_dir,
                        save_format=args.save_format,
                        random_kernels=args.random_kernels,
                        random_threshold_scale=args.random_threshold_scale,
                    )
                else:
                    missing_ckpts.append((name, matched_run))
                    print(f"Warning: No checkpoint found for '{name}' in {matched_run}")
            else:
                missing_folders.append(name)
                print(f"Warning: No folder matching '{name}' found in {args.root_dir}")

        if missing_folders or missing_ckpts:
            print("\n" + "=" * 60)
            print("SUMMARY OF UNRESOLVED MATCHES")
            print("=" * 60)
            if missing_folders:
                print(f"\nFolders not found ({len(missing_folders)}):")
                for name in missing_folders:
                    print(f"  - {name}")
            if missing_ckpts:
                print(f"\nCheckpoints not found ({len(missing_ckpts)}):")
                for name, matched_run in missing_ckpts:
                    print(f"  - {name} (matched folder: {matched_run})")
            print("=" * 60 + "\n")

    else:
        # Default behavior - find latest available run matching glob pattern
        runs = glob.glob(args.glob_pattern)
        runs.sort(reverse=True)

        experiment = None
        ckpt_name = None
        for run in runs:
            cfg_path = os.path.join(run, "configs/parsed.yaml")
            if os.path.exists(cfg_path):
                ckpt_dir = os.path.join(run, "ckpts")
                if os.path.exists(ckpt_dir):
                    pts = glob.glob(os.path.join(ckpt_dir, "*.pt")) + glob.glob(
                        os.path.join(ckpt_dir, "*.npz")
                    )
                    if len(pts) > 0:
                        experiment = run
                        ckpt_name = pts[0]
                        break

        if experiment is None:
            print(
                "Error: No valid experiment run with a config and checkpoint was found."
            )
            sys.exit(1)
        else:
            print(f"Using experiment run: {experiment}")
            print(f"Using checkpoint: {ckpt_name}")

        render(
            experiment,
            ckpt_name,
            n_frames=args.n_frames,
            fps=args.fps,
            output_dir=None,
            mesh_dir=args.mesh_dir,
            save_format=args.save_format,
            random_kernels=args.random_kernels,
            random_threshold_scale=args.random_threshold_scale,
        )
