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

import optuna
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

from hktex.utils import mibitmaps2torch, compute_all_image_metrics
from optimisation import main


def trainable(config, root, all_filenames, resolver_paths):
    """
    Ray Tune trainable function.

    Args:
        config (dict): Hyperparameter configuration from Ray Tune.
        root (str): Root directory for the dataset.
        all_filenames (list): List of all mesh filenames to process.
        resolver_paths (list[str]): List of paths for the Mitsuba FileResolver.
    """
    # Set the variant and add the necessary paths to the FileResolver in the worker.
    # This is crucial for Mitsuba to find its plugins in a distributed environment.
    mi.set_variant("cuda_ad_rgb")
    resolver = mi.Thread.thread().file_resolver()
    for path in resolver_paths:
        if path not in resolver:
            resolver.append(path)

    # Get the absolute path to the project root to resolve config file paths
    project_root = Path(__file__).resolve().parent.parent

    args_dict = {
        "config": str(project_root / "configs/texture_hktex_knn.yaml"),
        "rendering_config": str(project_root / "configs/render.yaml"),
        "verbose": False,
    }
    args = argparse.Namespace(**args_dict)

    total_iters = config["optim.iters"]

    # --- Dynamically build optimizer config ---
    optimizers_config = []
    adam_optim = {
        "name": "Adam",
        "args": {},
        "params": {
            "model._mean_colour": {"lr": config["lr_mean_colour"]},
            "model._kernel_colours": {"lr": config["lr_colours"]},
            "model._angles": {"lr": config["lr_angles"]},
            "model._anisotropies": {"lr": config["lr_anisotropies"]},
            "model._thresholds": {"lr": config["lr_thresholds"]},
            "model._sharpnesses": {"lr": config["lr_sharpnesses"]},
        },
    }
    if config["adam_scheduler"] == "cosine":
        adam_optim["scheduler"] = {
            "name": "CosineAnnealingLR",
            "args": {"T_max": total_iters, "eta_min": 1e-7},
        }
    elif config["adam_scheduler"] == "step":
        adam_optim["scheduler"] = {
            "name": "StepLR",
            "args": {"step_size": 1000, "gamma": 0.5},
        }
    else:
        pass  # No scheduler
    optimizers_config.append(adam_optim)

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
            "args": {"T_max": total_iters, "eta_min": 1e-7},
        }
    elif config["geodesic_scheduler"] == "step":
        geodesic_optim["scheduler"] = {
            "name": "StepLR",
            "args": {"step_size": 1000, "gamma": 0.5},
        }
    else:
        pass  # No scheduler
    optimizers_config.append(geodesic_optim)

    # Convert the Python object to a YAML string
    config["trainer.optimizers"] = yaml.dump(
        optimizers_config, default_flow_style=False
    )

    # --- Dynamically build density controller config ---
    # Scale iteration-based parameters proportionally to the total iterations

    # Ensure error_accumulation_interval is less than densify_interval
    has_error_based = "dc_densify_interval_frac" in config
    has_prune = "dc_importance_prune_interval_frac" in config
    density_controller_str = ""
    if not has_error_based and not has_prune:
        density_controller_str = "[]"
    if has_error_based:
        densify_interval = int(total_iters * config["dc_densify_interval_frac"])
        error_accumulation_interval = int(
            densify_interval * config["dc_error_accum_ratio"]
        )
        density_controller_str += f"""
        - density_controller_type: density_controllers.error_based_densification
          args:
            start_iter: {int(total_iters * config["dc_error_start_iter_frac"])}
            densify_interval: {densify_interval}
            error_accumulation_interval: {error_accumulation_interval}
            error_threshold: {config["dc_error_threshold"]}
            size_threshold: {config["dc_size_threshold"]}
            split_radius: {config["dc_error_split_radius"]}
            max_densify_ratio: {config["dc_max_densify_ratio"]}
            max_kernels: {min(config['dc_max_kernels_max'], config['dc_max_kernels_mult'] * config["trainer.model.n_sources"])}
            stop_iter: {int(total_iters * config["dc_error_stop_iter_frac"])}
        """
    if has_prune:
        prune_interval = int(total_iters * config["dc_importance_prune_interval_frac"])
        prune_accumulation_interval = int(
            prune_interval * config["dc_importance_accum_ratio"]
        )
        density_controller_str += f"""
        - density_controller_type: density_controllers.importance_pruning
          args:
            start_iter: {int(total_iters * config["dc_importance_start_iter_frac"])}
            prune_interval: {prune_interval}
            accumulation_interval: {prune_accumulation_interval}
            selection_threshold: {config["dc_importance_selection_threshold"]}
            contrib_threshold: {config["dc_importance_contrib_threshold"]}
            stop_iter: {int(total_iters * config["dc_importance_stop_iter_frac"])}
        """

    config["trainer.density_controllers"] = density_controller_str

    # Remove the temporary keys from the config before passing to main
    del config["adam_scheduler"]
    del config["geodesic_scheduler"]
    del config["momentum_locations"]
    for k in list(config.keys()):
        if k.startswith("dc_") or k.startswith("lr_"):
            del config[k]

    # Get the absolute path to the current trial directory from Ray Tune
    # and use it as the root for all outputs from the optimization script.
    config["exp_root_dir"] = tune.get_context().get_trial_dir()

    # The main loop over files is now inside the trainable
    per_mesh_errors = []
    per_mesh_metrics = []
    per_mesh_kernels = []

    # Wrap the file loop in tqdm for a progress bar within each trial
    for fname in tqdm(all_filenames, desc=f"Trial files", leave=False):
        # Create a copy of the config and update the mesh path for the current file
        current_config = config.copy()
        current_config["data.mesh_path"] = os.path.join(root, fname)

        # Convert the config dict to the format expected by `main`
        extras = [f"{k}={v}" for k, v in current_config.items()]

        try:
            out = main(args, extras)
            gt_rend, result_rend, _, _ = out["renderings"]

            # Calculate error for the current mesh
            gt = mibitmaps2torch(gt_rend)
            res = mibitmaps2torch(result_rend)
            error = F.mse_loss(res, gt, reduction="mean")
            metrics = compute_all_image_metrics(res, gt)
            per_mesh_errors.append(error)
            per_mesh_metrics.append(metrics)
            per_mesh_kernels.append(
                out["optimisation"].model._kernel_locations.shape[0]
            )
        except Exception as e:
            print(f"Error processing {fname}:")
            traceback.print_exc()
            # Penalize failures heavily
            per_mesh_errors.append(10.0)

    # Report the average error across all files as the final metric
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
            "storage_torch_kb": out["storage"][0],
            "storage_npz_kb": out["storage"][1],
            **avg_metrics,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for HKTex.")
    parser.add_argument(
        "--root",
        type=str,
        default="/data2/home/objaverse",
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--optim_iters",
        type=int,
        default=100,
        help="Number of optimization iterations.",
    )
    parser.add_argument(
        "--max_kernels",
        type=int,
        default=50_000,
        help="Maximum number of kernels allowed.",
    )
    parser.add_argument(
        "--run_id", type=str, default="tune_test", help="Identifier for the tuning run."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of hyperparameter combinations to try.",
    )
    parser.add_argument(
        "--max_concurrent_trials",
        type=int,
        default=4,
        help="Maximum number of trials to run concurrently.",
    )
    parser.add_argument(
        "--gpus_per_trial",
        type=float,
        default=1.0,
        help="Number of GPUs to allocate per trial (can be fractional, e.g. 0.5).",
    )

    cli_args = parser.parse_args()

    root = cli_args.root
    subset_filenames = [
        # "../objects/spot/spot_triangulated.obj",
        os.path.join(root, "000-087/0e708d1e0ce0447ba5637a5320f5729c.glb"),  # octopus
        os.path.join(root, "000-018/998d641ce1c74e44978a91fedc849905.glb"),  # smokepipe
        os.path.join(root, "000-074/5ecf9d1175ae405a9a073db305786411.glb"),  # barrel
        # os.path.join(root, "000-101/818e088dc59f4a89bfea14cb46a4beca.glb"),  # wmelon
        # os.path.join(root, "000-138/6713cc0cdad34f89a0256c5d2f68b7c1.glb"),  # turtle
        # os.path.join(root, "000-013/d79a32a512c64c5e93dc856864789a7e.glb"),  # halloween
        os.path.join(root, "000-096/db5f9c28708142909b15212625a127f9.glb"),  # ball
        os.path.join(root, "000-066/e7caba92073d4adba3477c21aa25e91f.glb"),  # vase
    ]

    search_space = {
        # Fixed parameters #############################################################
        "name": "output_experiment",
        "trainer.tracer.debug": False,
        "optim.iters": cli_args.optim_iters,
        "dc_max_kernels_max": cli_args.max_kernels,
        "renderer.n_rotating_frames": 3,
        "trainer.model.mass_type": "one",
        "trainer.eigen_albo.parse_weighting_from_str": True,
        "trainer.model.allow_negative_colours": True,
        # "trainer.model.knn_inner_k": None,
        #
        # Tunable parameters ###########################################################
        "trainer.model.n_sources": tune.choice(
            [1000, 2500, 5000]
        ),  # tune.choice([1000, 2500, 5000, 10000, 20000]),
        "data.batch_size": tune.choice([512, 1024, 2048]),
        "data.use_importance_sampling": tune.choice([True, False]),
        "data.importance_sampling_pool_size": tune.choice(
            [1_000_000, 5_000_000, 10_000_000]
        ),
        "data.importance_sampling_warmup_steps": tune.choice([250, 500, 1_000]),
        "data.importance_sampling_ema_beta": tune.choice([0.9, 0.95, 0.99]),
        "trainer.eigen_albo.local_frames": "principal_curvatures",  # tune.choice(
        # ["principal_curvatures", "axis_aligned_20", "axis_aligned_5"]
        # ),
        "trainer.model.diff_time": tune.loguniform(1e-6, 0.2),
        "trainer.eigen_albo.distance_weighting": tune.choice(
            ["none", "gaussian_0.1", "gaussian_0.05", "inverse"]
        ),
        "trainer.model.init_min_threshold": tune.choice([0.3, 0.5, 0.9, 0.999]),
        "trainer.model.range_enforcement_type": "pgd",  # tune.choice(["pgd", "activations"]),
        "trainer.model.init_kernel_edge_type": tune.choice(["uniform", "high_skewed"]),
        # "trainer.model.allow_negative_colours": tune.choice([True, False]),
        "trainer.loss_type": tune.choice(["mse_loss", "smooth_l1_loss", "l1_loss"]),
        "trainer.data_initialisation_random_ratio": tune.choice([0.0, 0.5, 1.0]),
        "adam_scheduler": tune.choice(["none", "step", "cosine"]),
        "geodesic_scheduler": tune.choice(["none", "step", "cosine"]),
        "lr_mean_colour": tune.loguniform(1e-5, 1e-2),
        "lr_colours": tune.loguniform(1e-4, 1e-1),
        "lr_angles": tune.loguniform(1e-4, 1e-1),
        "lr_anisotropies": tune.loguniform(1e-4, 1e-1),
        "lr_thresholds": tune.loguniform(1e-4, 1e-1),
        "lr_sharpnesses": tune.loguniform(1e-4, 1e-1),
        "lr_locations": tune.loguniform(1e-4, 1e-1),
        "momentum_locations": tune.choice([0.0, 0.7, 0.9]),
        "dc_importance_selection_threshold": tune.loguniform(0.001, 0.05),
        "dc_importance_contrib_threshold": tune.loguniform(0.001, 0.1),
        # "dc_error_threshold": tune.loguniform(0.005, 0.05),
        # "dc_size_threshold": tune.uniform(0.1, 0.4),
        # "dc_max_densify_ratio": tune.choice([0.3, 0.4, 0.5]),
        "dc_importance_start_iter_frac": tune.choice([0.1, 0.2]),
        "dc_importance_prune_interval_frac": tune.choice([0.1, 0.2, 0.5]),
        "dc_importance_accum_ratio": tune.choice([0.25, 0.5, 0.75]),
        "dc_importance_stop_iter_frac": tune.choice([0.5, 0.6, 0.7]),
        # "dc_error_start_iter_frac": tune.choice([0.05, 0.08]),
        # "dc_densify_interval_frac": tune.choice([0.01, 0.02, 0.04, 0.08]),
        # "dc_error_accum_ratio": tune.choice([0.25, 0.5, 0.75]),
        # "dc_error_stop_iter_frac": tune.choice([0.7, 0.8]),
        # "dc_error_split_radius": tune.choice([0.1, 0.15, 0.2]),
        # "dc_max_kernels_mult": tune.choice([2, 5, 10]),
    }

    search_alg = OptunaSearch(
        metric=["mean_error", "storage_npz_kb"],
        mode=["min", "min"],
    )
    if cli_args.max_concurrent_trials > 0:
        search_alg = ConcurrencyLimiter(
            search_alg, max_concurrent=cli_args.max_concurrent_trials
        )

    # Set the variant in the main process and capture the FileResolver paths.
    # This list of paths is serializable and can be sent to Ray workers.
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
        param_space=search_space,
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
    print("TUNING COMPLETE")
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

    # Get a pandas DataFrame with the results
    df = results.get_dataframe()

    # Define the columns you are interested in seeing
    # We'll show the error and the hyperparameters we tuned
    hparam_cols = list(search_space.keys())
    # Filter out fixed parameters
    hparam_cols = [
        h for h in hparam_cols if isinstance(search_space[h], tune.search.sample.Domain)
    ]

    display_cols = ["mean_error", "storage_npz_kb"] + [
        f"config/{h}" for h in hparam_cols
    ]

    # Sort by the metric and show the top 5 trials
    sorted_df = df.sort_values("mean_error", ascending=True)
    print("\nTop 5 Best Hyperparameter Configurations:")
    print(sorted_df[display_cols].head(5).to_string())

    # --- Generate and save Optuna visualization plots ---
    # The Optuna study object is stored within the search_alg
    study = search_alg.searcher._ot_study
    results_path = results.get_best_result().path
    experiment_dir = Path(results_path).parent

    print(f"\nSaving Optuna plots to: {experiment_dir}")

    try:
        # Plot optimization history
        history_plot = optuna.visualization.plot_optimization_history(study)
        history_plot.write_html(os.path.join(experiment_dir, "optuna_history.html"))

        # Plot parameter importances
        importance_plot = optuna.visualization.plot_param_importances(study)
        importance_plot.write_html(
            os.path.join(experiment_dir, "optuna_importances.html")
        )

        # Plot slice plot to see parameter relationships
        slice_plot = optuna.visualization.plot_slice(study)
        slice_plot.write_html(os.path.join(experiment_dir, "optuna_slice.html"))

        # Plot Pareto front for multi-objective optimization
        pareto_plot = optuna.visualization.plot_pareto_front(
            study, target_names=["mean_error", "storage_npz_kb"]
        )
        pareto_plot.write_html(os.path.join(experiment_dir, "optuna_pareto.html"))

        print("Successfully saved Optuna plots.")

    except Exception as e:
        print(f"\nAn error occurred while generating Optuna plots: {e}")
