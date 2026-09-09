import sys
import yaml
from pathlib import Path
import os

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import argparse
import traceback
import torch
import torch.nn.functional as F
from tqdm import tqdm
import drjit as dr

import optuna
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

from hktex.utils import mibitmaps2torch, compute_all_image_metrics
from optimisation import main


def define_optuna_space(trial: optuna.Trial) -> dict:
    constants = {}
    # Data sampling
    use_importance = trial.suggest_categorical(
        "data.use_importance_sampling", [False, True]
    )
    if use_importance:
        trial.suggest_categorical("data.importance_sampling_warmup_steps", [250, 500])
    else:
        constants["data.importance_sampling_warmup_steps"] = 0

    # Reconstruction objective
    trial.suggest_categorical(
        "trainer.loss_type", ["mse_loss", "smooth_l1_loss", "l1_loss"]
    )

    # Kernel interpolation
    trial.suggest_categorical("trainer.model.knn_k", [10, 12, 16, 20, 24])
    trial.suggest_categorical("trainer.model.softmax_temperature", [0.015, 0.02, 0.03])
    trial.suggest_categorical(
        "trainer.model.softmax_temperature_min", [2e-4, 5e-4, 1e-3, 2e-3]
    )

    # Residual composition / recentering
    trial.suggest_categorical("trainer.model.residual_gain_init", [0.8, 1.0])
    residual_recenter_enabled = trial.suggest_categorical(
        "residual_recenter_enabled", [False, True]
    )
    if residual_recenter_enabled:
        trial.suggest_categorical(
            "trainer.model.residual_recenter_every", [100, 200, 400]
        )
        trial.suggest_categorical("residual_recenter_stop_iter_frac", [0.6, 0.8])
    else:
        constants["trainer.model.residual_recenter_every"] = None
        constants["residual_recenter_stop_iter_frac"] = 0.8

    # Regularisation weights (true conditional)
    tv_enabled = trial.suggest_categorical("trainer.tv_loss_enabled", [False, True])
    if tv_enabled:
        trial.suggest_float("trainer.tv_loss_weight", 1e-6, 5e-4, log=True)
    else:
        constants["trainer.tv_loss_weight"] = 0.0

    smooth_enabled = trial.suggest_categorical(
        "trainer.smoothness_loss_enabled", [False, True]
    )
    if smooth_enabled:
        trial.suggest_float("trainer.smoothness_loss_weight", 1e-6, 2e-4, log=True)
    else:
        constants["trainer.smoothness_loss_weight"] = 0.0

    # Optimizer schedules
    trial.suggest_categorical("adam_scheduler", ["cosine", "step"])
    trial.suggest_categorical("geodesic_scheduler", ["step", "cosine"])

    # Adam learning rates
    trial.suggest_float("lr_mean_colour", 5e-4, 3e-3, log=True)
    trial.suggest_float("lr_colours", 3e-3, 1e-1, log=True)
    trial.suggest_float("lr_tau", 2e-5, 1e-3, log=True)
    trial.suggest_float("lr_residual_gain", 5e-5, 5e-3, log=True)

    # Geodesic optimizer
    trial.suggest_float("lr_locations", 1e-4, 2e-2, log=True)
    trial.suggest_categorical("momentum_locations", [0.0, 0.8, 0.9])

    # Define-by-run pattern: suggest_* populates trial params.
    return constants


def trainable(config, root, all_filenames, resolver_paths):
    """
    Ray Tune trainable function for PCL UV texture fitting.
    """
    mi.set_variant("cuda_ad_rgb")
    resolver = mi.Thread.thread().file_resolver()
    for path in resolver_paths:
        if path not in resolver:
            resolver.append(path)

    project_root = Path(__file__).resolve().parent.parent
    args_dict = {
        "config": str(project_root / "configs/uv_texture_fitting_pcl.yaml"),
        "rendering_config": str(project_root / "configs/rendering.yaml"),
        "verbose": False,
    }
    args = argparse.Namespace(**args_dict)

    total_iters = int(config["optim.iters"])
    stop_iter = int(total_iters * float(config["residual_recenter_stop_iter_frac"]))

    adam_optim = {
        "name": "Adam",
        "args": {},
        "params": {
            "model._mean_colour": {"lr": config["lr_mean_colour"]},
            "model._kernel_colours": {"lr": config["lr_colours"]},
            "model._softmax_temperature_raw": {"lr": config["lr_tau"]},
            "model._residual_gain_raw": {"lr": config["lr_residual_gain"]},
        },
    }
    if config["adam_scheduler"] == "cosine":
        adam_optim["scheduler"] = {
            "name": "CosineAnnealingLR",
            "args": {"T_max": total_iters, "eta_min": 1e-6},
        }
    elif config["adam_scheduler"] == "step":
        adam_optim["scheduler"] = {
            "name": "StepLR",
            "args": {"step_size": 1200, "gamma": 0.7},
        }

    geodesic_optim = {
        "name": "GeodesicOpt",
        "args": {
            "lr": config["lr_locations"],
            "momentum": config["momentum_locations"],
        },
        "tracer": "tracer",
        "params": {"model._kernel_locations": {"face_ids": "model._kernel_face_ids"}},
    }
    if config["geodesic_scheduler"] == "cosine":
        geodesic_optim["scheduler"] = {
            "name": "CosineAnnealingLR",
            "args": {"T_max": total_iters, "eta_min": 1e-6},
        }
    elif config["geodesic_scheduler"] == "step":
        geodesic_optim["scheduler"] = {
            "name": "StepLR",
            "args": {"step_size": 1200, "gamma": 0.7},
        }

    config["trainer.optimizers"] = yaml.dump(
        [adam_optim, geodesic_optim], default_flow_style=False
    )

    # Remove helper-only keys from CLI overrides.
    for k in (
        "adam_scheduler",
        "geodesic_scheduler",
        "momentum_locations",
        "lr_mean_colour",
        "lr_colours",
        "lr_tau",
        "lr_residual_gain",
        "lr_locations",
        "residual_recenter_stop_iter_frac",
        "residual_recenter_enabled",
        "trainer.tv_loss_enabled",
        "trainer.smoothness_loss_enabled",
    ):
        del config[k]

    config["trainer.model.residual_recenter_stop_iter"] = stop_iter
    config["exp_root_dir"] = tune.get_context().get_trial_dir()

    per_mesh_errors = []
    per_mesh_metrics = []
    per_mesh_kernels = []
    storage_torch_kb = float("inf")
    storage_npz_kb = float("inf")

    for fname in tqdm(all_filenames, desc="Trial files", leave=False):
        current_config = config.copy()
        current_config["data.mesh_path"] = os.path.join(root, fname)
        extras = [f"{k}={v}" for k, v in current_config.items()]
        out = None
        gt_rend = result_rend = None
        gt = res = error = metrics = None

        try:
            out = main(args, extras)
            gt_rend, result_rend, _, _ = out["renderings"]
            gt = mibitmaps2torch(gt_rend)
            res = mibitmaps2torch(result_rend)
            error = F.mse_loss(res, gt, reduction="mean")
            metrics = compute_all_image_metrics(res, gt)
            per_mesh_errors.append(error)
            per_mesh_metrics.append(metrics)
            per_mesh_kernels.append(
                out["optimisation"].model._kernel_locations.shape[0]
            )
            storage_torch_kb, storage_npz_kb = out["storage"]
        except Exception:
            print(f"Error processing {fname}:")
            traceback.print_exc()
            per_mesh_errors.append(10.0)
        finally:
            # Free FAISS database/index allocations from the previous mesh run when available.
            if out is not None and "optimisation" in out:
                optim = out["optimisation"]
                texture = getattr(optim, "model", None)
                # In this tuner we optimize PCLTexture directly.
                # Fallback handles wrapped models if used elsewhere.
                if texture is not None and not hasattr(texture, "_faiss_index"):
                    texture = getattr(texture, "model", None)

                faiss_index = getattr(texture, "_faiss_index", None)
                if faiss_index is not None and hasattr(faiss_index, "reset"):
                    faiss_index.reset()
                    if hasattr(texture, "mark_knn_dirty"):
                        texture.mark_knn_dirty()

            # Release temporary per-mesh objects and flush alloc caches.
            del out, current_config, extras
            del gt_rend, result_rend, gt, res, error, metrics
            dr.flush_malloc_cache()
            torch.cuda.empty_cache()

    n_success = len(per_mesh_metrics)
    n_failed = len(all_filenames) - n_success
    if n_success == 0:
        tune.report(
            {
                "mean_error": 10.0,
                "std_error": 0.0,
                "mean_n_kernels": 0.0,
                "n_success": 0,
                "n_failed": n_failed,
                "storage_torch_kb": 1e12,
                "storage_npz_kb": 1e12,
            }
        )
        return

    per_mesh_errors = torch.tensor(per_mesh_errors)
    avg_error = per_mesh_errors.mean().item()
    std_error = per_mesh_errors.std().item()
    avg_n_kernels = sum(per_mesh_kernels) / len(per_mesh_kernels)
    avg_metrics = {
        "avg_" + k: sum(m[k] for m in per_mesh_metrics) / len(per_mesh_metrics)
        for k in per_mesh_metrics[0]
    }
    tune.report(
        {
            "mean_error": avg_error,
            "std_error": std_error,
            "mean_n_kernels": avg_n_kernels,
            "n_success": n_success,
            "n_failed": n_failed,
            "storage_torch_kb": storage_torch_kb,
            "storage_npz_kb": storage_npz_kb,
            **avg_metrics,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for PCL textures."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/data2/home/sf3018/objaverse",
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="tune_pcl",
        help="Identifier for the tuning run.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of hyperparameter combinations to try.",
    )
    parser.add_argument(
        "--max_concurrent_trials",
        type=int,
        default=2,
        help="Maximum number of trials to run concurrently.",
    )
    parser.add_argument(
        "--gpus_per_trial",
        type=float,
        default=1.0,
        help="Number of GPUs to allocate per trial.",
    )
    parser.add_argument(
        "--n_sources",
        type=int,
        default=10000,
        choices=[10000, 50000],
        help="Fixed number of PCL sources for this tuning run.",
    )
    parser.add_argument(
        "--optim_iters",
        type=int,
        default=10000,
        help="Number of optimization iterations.",
    )
    cli_args = parser.parse_args()

    root = cli_args.root
    subset_filenames = [
        os.path.join(root, "000-087/0e708d1e0ce0447ba5637a5320f5729c.glb"),  # octopus
        os.path.join(root, "000-018/998d641ce1c74e44978a91fedc849905.glb"),  # smokepipe
        os.path.join(root, "000-074/5ecf9d1175ae405a9a073db305786411.glb"),  # barrel
        os.path.join(root, "000-096/db5f9c28708142909b15212625a127f9.glb"),  # ball
        os.path.join(root, "000-066/e7caba92073d4adba3477c21aa25e91f.glb"),  # vase
    ]

    fixed_params = {
        # Fixed parameters: experiment/runtime
        "name": f"tune_pcl_{cli_args.n_sources}",
        "optim.iters": int(cli_args.optim_iters),
        "renderer.n_rotating_frames": 3,
        "trainer.tracer.debug": False,
        # Fixed parameters: trainer/module wiring
        "trainer.use_knn_implementation": True,
        "trainer.model_type": "modules.pcl-texture",
        "trainer.data_initialisation_random_ratio": 1.0,
        # Fixed parameters: data sampling
        "data.sampling_method": "uniform",
        "data.importance_sampling_pool_size": 1_000_000,
        "data.importance_sampling_ema_beta": 0.99,
        # Fixed parameters: model structure
        "trainer.model.n_sources": int(cli_args.n_sources),
        "trainer.model.out_net": False,
        "trainer.model.normalize_colours": False,
        "trainer.model.allow_negative_colours": True,
        "trainer.model.range_enforcement_type": "pgd",
        # Fixed parameters: interpolation kernel
        "trainer.model.weighting": "softmax_rbf",
        "trainer.model.distance_eps": 1e-8,
        "trainer.model.use_face_normal_weighting": False,
        "trainer.model.normal_weight_beta": 8.0,
        # Fixed parameters: FAISS
        "trainer.model.faiss.use_float16": False,
        "trainer.model.faiss.distance_impl": "bmm",
        "trainer.model.faiss.build_db_norm": True,
        "trainer.model.faiss.compile_distances": True,
        # Fixed parameters: residual bounds + optional features
        "trainer.model.residual_gain_min": 0.0,
        "trainer.edge_aware_photo_enabled": False,
        "trainer.edge_aware_reg_enabled": False,
        "trainer.density_controllers": [],
        # Fixed for now from current findings
        "data.batch_size": 4096,
    }

    search_alg = OptunaSearch(
        space=define_optuna_space,
        metric=["mean_error", "storage_npz_kb"],
        mode=["min", "min"],
    )
    if cli_args.max_concurrent_trials > 0:
        search_alg = ConcurrencyLimiter(
            search_alg, max_concurrent=cli_args.max_concurrent_trials
        )

    mi.set_variant("cuda_ad_rgb")
    resolver_paths = [str(p) for p in list(mi.Thread.thread().file_resolver())]

    trainable_with_gpu = tune.with_resources(
        tune.with_parameters(
            trainable,
            root=root,
            all_filenames=subset_filenames,
            resolver_paths=resolver_paths,
        ),
        {"gpu": cli_args.gpus_per_trial},
    )

    tuner = tune.Tuner(
        trainable_with_gpu,
        param_space=fixed_params,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            num_samples=cli_args.num_samples,
        ),
        run_config=tune.RunConfig(
            storage_path=os.path.abspath("outputs/ray_tune"), name=cli_args.run_id
        ),
    )
    results = tuner.fit()

    print("\n" + "=" * 80)
    print("PCL TUNING COMPLETE")
    print("=" * 80)

    best_error_result = results.get_best_result(metric="mean_error", mode="min")
    print("Best error trial config: ", best_error_result.config)
    print(
        f"Best trial final validation loss: {best_error_result.metrics['mean_error']:.4f}"
    )
    print(
        f"Corresponding model size: {best_error_result.metrics['storage_npz_kb']:.2f} KB"
    )

    best_size_result = results.get_best_result(metric="storage_npz_kb", mode="min")
    print("Best size trial config: ", best_size_result.config)
    print(
        f"Best trial final model size: {best_size_result.metrics['storage_npz_kb']:.2f} KB"
    )
    print(
        f"Corresponding validation loss: {best_size_result.metrics['mean_error']:.4f}"
    )

    df = results.get_dataframe()
    hparam_cols = [
        "data.use_importance_sampling",
        "data.importance_sampling_warmup_steps",
        "trainer.loss_type",
        "trainer.model.knn_k",
        "trainer.model.softmax_temperature",
        "trainer.model.softmax_temperature_min",
        "trainer.model.residual_gain_init",
        "residual_recenter_enabled",
        "trainer.model.residual_recenter_every",
        "residual_recenter_stop_iter_frac",
        "trainer.tv_loss_enabled",
        "trainer.tv_loss_weight",
        "trainer.smoothness_loss_enabled",
        "trainer.smoothness_loss_weight",
        "adam_scheduler",
        "geodesic_scheduler",
        "lr_mean_colour",
        "lr_colours",
        "lr_tau",
        "lr_residual_gain",
        "lr_locations",
        "momentum_locations",
    ]
    display_cols = ["mean_error", "storage_npz_kb"] + [
        f"config/{h}" for h in hparam_cols
    ]
    sorted_df = df.sort_values("mean_error", ascending=True)
    print("\nTop 5 Best Hyperparameter Configurations:")
    print(sorted_df[display_cols].head(5).to_string())

    study = search_alg.searcher._ot_study
    results_path = best_error_result.path
    experiment_dir = Path(results_path).parent
    print(f"\nSaving Optuna plots to: {experiment_dir}")

    try:
        mean_error_target = lambda t: t.values[0]
        history_plot = optuna.visualization.plot_optimization_history(
            study,
            target=mean_error_target,
            target_name="mean_error",
        )
        history_plot.write_html(os.path.join(experiment_dir, "optuna_history.html"))

        importance_plot = optuna.visualization.plot_param_importances(
            study,
            target=mean_error_target,
            target_name="mean_error",
        )
        importance_plot.write_html(
            os.path.join(experiment_dir, "optuna_importances.html")
        )

        slice_plot = optuna.visualization.plot_slice(
            study,
            target=mean_error_target,
            target_name="mean_error",
        )
        slice_plot.write_html(os.path.join(experiment_dir, "optuna_slice.html"))

        pareto_plot = optuna.visualization.plot_pareto_front(
            study, target_names=["mean_error", "storage_npz_kb"]
        )
        pareto_plot.write_html(os.path.join(experiment_dir, "optuna_pareto.html"))
        print("Successfully saved Optuna plots.")
    except Exception as e:
        print(f"\nAn error occurred while generating Optuna plots: {e}")
