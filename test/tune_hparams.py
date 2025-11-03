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

import torch.nn.functional as F
from tqdm import tqdm

import optuna
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter

import mitsuba as mi

from heatsplats.utils import mibitmaps2torch
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
        "config": str(project_root / "configs/uv_texture_fitting.yaml"),
        "rendering_config": str(project_root / "configs/rendering.yaml"),
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
            "model._kernel_colours": {"lr": config["lr_colours"]},
            "model._angles": {"lr": config["lr_angles"]},
            "model._anisotropies": {"lr": config["lr_anisotropies"]},
            "model._thresholds": {"lr": config["lr_thresholds"]},
            "model._sharpnesses": {"lr": config["lr_sharpnesses"]},
            "model._opacities": {"lr": config["lr_opacities"]},
        },
    }
    if config["use_adam_scheduler"]:
        adam_optim["scheduler"] = {
            "name": "StepLR",
            "args": {"step_size": 1000, "gamma": 0.5},
        }
    optimizers_config.append(adam_optim)

    geodesic_optim = {
        "name": "GeodesicOpt",
        "args": {"lr": config["lr_locations"]},
        "tracer": "tracer",
        "params": {"model._kernel_locations": {"face_ids": "model._kernel_face_ids"}},
    }
    if config["use_geodesic_scheduler"]:
        geodesic_optim["scheduler"] = {
            "name": "CosineAnnealingLR",
            "args": {"T_max": total_iters, "eta_min": 1e-7},
        }
    optimizers_config.append(geodesic_optim)

    # Convert the Python object to a YAML string
    config["trainer.optimizers"] = yaml.dump(
        optimizers_config, default_flow_style=False
    )

    # --- Dynamically build density controller config ---
    # Scale iteration-based parameters proportionally to the total iterations

    # Ensure error_accumulation_interval is less than densify_interval
    densify_interval = int(total_iters * config["dc_densify_interval_frac"])
    error_accumulation_interval = int(densify_interval * config["dc_error_accum_ratio"])

    config[
        "trainer.density_controllers"
    ] = f"""
    - density_controller_type: density_controllers.opacity
      args:
        start_iter: {int(total_iters * config["dc_opacity_start_iter_frac"])}
        prune_opacity: {config["dc_prune_opacity"]}
        prune_interval: {int(total_iters * config["dc_opacity_prune_interval_frac"])}
        reset_opacity_interval: {int(total_iters * config["dc_opacity_reset_interval_frac"])}
        stop_iter: {int(total_iters * config["dc_opacity_stop_iter_frac"])}
    - density_controller_type: density_controllers.error_based_densification
      args:
        start_iter: {int(total_iters * config["dc_error_start_iter_frac"])}
        densify_interval: {densify_interval}
        error_accumulation_interval: {error_accumulation_interval}
        error_threshold: {config["dc_error_threshold"]}
        size_threshold: {config["dc_size_threshold"]}
        max_densify_ratio: {config["dc_max_densify_ratio"]}
        max_kernels: {config["dc_max_kernels"]}
        stop_iter: {int(total_iters * config["dc_error_stop_iter_frac"])}
    """

    # Remove the temporary keys from the config before passing to main
    del config["use_adam_scheduler"]
    del config["use_geodesic_scheduler"]
    for k in list(config.keys()):
        if k.startswith("dc_") or k.startswith("lr_"):
            del config[k]

    # Get the absolute path to the current trial directory from Ray Tune
    # and use it as the root for all outputs from the optimization script.
    config["exp_root_dir"] = tune.get_context().get_trial_dir()

    # The main loop over files is now inside the trainable
    total_error = 0.0
    num_files = len(all_filenames)

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
            error = F.mse_loss(res, gt, reduction="mean").item()
            total_error += error
        except Exception as e:
            print(f"Error processing {fname}:")
            traceback.print_exc()
            # Penalize failures heavily
            total_error += 10.0

    # Report the average error across all files as the final metric
    avg_error = total_error / num_files
    tune.report({"mean_error": avg_error})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for GeoSplat.")
    parser.add_argument(
        "--root",
        type=str,
        default="/data2/objaverse/hf-objaverse-v1/glbs",
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
        default=10_000,
        help="Maximum number of kernels allowed.",
    )
    parser.add_argument(
        "--run_id", type=str, default="tune_0", help="Identifier for the tuning run."
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
        "name": "uv-hparam-search",
        "trainer.tracer.debug": False,
        "optim.iters": cli_args.optim_iters,
        "dc_max_kernels": cli_args.max_kernels,
        "renderer.n_rotating_frames": 3,
        #
        # Tunable parameters ###########################################################
        "trainer.model.n_sources": tune.choice([100, 500, 1000, 2000]),
        "data.batch_size": tune.choice([512, 1024]),
        "trainer.eigen_albo.local_frames": tune.choice(
            ["principal_curvatures", "axis_aligned_20", "axis_aligned_5"]
        ),
        "trainer.model.mass_type": tune.choice(["one", "kde", "interpolated"]),
        "trainer.model.diff_time": tune.loguniform(1e-8, 1e-1),
        "trainer.eigen_albo.distance_weighting": tune.choice(
            ["none", "gaussian_0.01", "gaussian_0.05", "inverse"]
        ),
        "trainer.model.init_min_threshold": tune.choice([0.3, 0.5, 0.9, 0.999]),
        "use_adam_scheduler": tune.choice([True, False]),
        "use_geodesic_scheduler": tune.choice([True, False]),
        "lr_colours": tune.loguniform(1e-3, 1e-1),
        "lr_angles": tune.loguniform(1e-5, 1e-3),
        "lr_anisotropies": tune.loguniform(1e-5, 1e-3),
        "lr_thresholds": tune.loguniform(1e-4, 1e-2),
        "lr_sharpnesses": tune.loguniform(1e-4, 1e-2),
        "lr_opacities": tune.loguniform(1e-4, 1e-2),
        "lr_locations": tune.loguniform(1e-4, 1e-1),
        "dc_prune_opacity": tune.loguniform(0.01, 0.1),
        "dc_error_threshold": tune.loguniform(0.01, 0.2),
        "dc_size_threshold": tune.uniform(0.1, 0.5),
        "dc_max_densify_ratio": tune.choice([0.2, 0.3, 0.4, 0.5]),
        "dc_opacity_start_iter_frac": tune.choice([0.1, 0.2]),
        "dc_opacity_prune_interval_frac": tune.choice([0.05, 0.1]),
        "dc_opacity_reset_interval_frac": tune.choice([0.1, 0.2]),
        "dc_opacity_stop_iter_frac": tune.choice([0.5, 0.6, 0.7]),
        "dc_error_start_iter_frac": tune.choice([0.02, 0.05, 0.08, 0.15]),
        "dc_densify_interval_frac": tune.choice([0.01, 0.02, 0.04, 0.08]),
        "dc_error_accum_ratio": tune.choice([0.25, 0.5, 0.75]),
        "dc_error_stop_iter_frac": tune.choice([0.7, 0.8, 0.9]),
    }

    search_alg = OptunaSearch()
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
        {"gpu": 1},
    )

    tuner = tune.Tuner(
        trainable_with_gpu,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="mean_error",
            mode="min",
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

    best_result = results.get_best_result()
    print("Best trial config: ", best_result.config)
    print(f"Best trial final validation loss: {best_result.metrics['mean_error']:.4f}")

    # Get a pandas DataFrame with the results
    df = results.get_dataframe()

    # Define the columns you are interested in seeing
    # We'll show the error and the hyperparameters we tuned
    hparam_cols = list(search_space.keys())
    # Filter out fixed parameters
    hparam_cols = [
        h for h in hparam_cols if isinstance(search_space[h], tune.search.sample.Domain)
    ]

    display_cols = ["mean_error"] + [f"config/{h}" for h in hparam_cols]

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

        print("Successfully saved Optuna plots.")

    except Exception as e:
        print(f"\nAn error occurred while generating Optuna plots: {e}")
