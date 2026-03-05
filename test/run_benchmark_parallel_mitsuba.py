import sys
import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
import random

# Fix for torch._dynamo recompilation issues with dynamic shapes
torch._dynamo.config.force_parameter_static_shapes = False
torch._dynamo.config.cache_size_limit = 128

# Add project root to path
try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

# Pre-import libraries to avoid C++ ABI conflicts
try:
    import faiss
    import faiss.contrib.torch_utils
except ImportError:
    pass

import mitsuba as mi
from ray import tune

from heatsplats.data.objaverse_downloader import find_filenames
from heatsplats.utils import mibitmaps2torch, compute_all_image_metrics
from optimisation import main


def benchmark_trainable(config):
    # Ensure imports are correct in worker to avoid ABI conflicts
    try:
        import faiss
        import faiss.contrib.torch_utils
    except ImportError:
        pass

    import mitsuba as mi
    import torch

    # Fix for torch._dynamo recompilation issues with dynamic shapes (changing kernel counts)
    torch._dynamo.config.force_parameter_static_shapes = False
    torch._dynamo.config.cache_size_limit = 128

    try:
        mi.set_variant("cuda_ad_rgb")
    except Exception:
        pass

    fname = config["filename"]
    root = config["root"]
    base_config_path = config["base_config_path"]
    rendering_config_path = config["rendering_config_path"]

    args = argparse.Namespace(
        config=base_config_path,
        rendering_config=rendering_config_path,
        gpu="0",
        verbose=False,
    )

    trial_dir = tune.get_context().get_trial_dir()
    mesh_path = os.path.join(root, fname)
    safe_name = fname.replace("/", "_").replace("\\", "_").replace(".glb", "")

    # Configure output to go into the trial directory
    extras = [
        f"data.mesh_path={mesh_path}",
        f"trainer.network.eigen_albo.mesh_path={mesh_path}",
        f"exp_root_dir={trial_dir}",
        "name=output",
        f"tag={safe_name}",
        "use_timestamp=False",
        "optim.save_model=True",
        "optim.save_logs=True",
    ]

    try:
        out = main(args, extras, render=True)

        gt_rend, result_rend, _, _ = out["renderings"]
        gt = mibitmaps2torch(gt_rend)
        res = mibitmaps2torch(result_rend)

        mse = F.mse_loss(res, gt, reduction="mean").item()
        metrics = compute_all_image_metrics(res, gt)

        storage_torch_kb = out["storage"][0]
        storage_npz_kb = out["storage"][1]

        n_kernels = 0
        if hasattr(out["optimisation"].model.model, "_kernel_locations"):
            n_kernels = out["optimisation"].model.model._kernel_locations.shape[0]
        elif hasattr(out["optimisation"].model.model, "N_sources"):
            n_kernels = out["optimisation"].model.model.N_sources

        row = {
            "filename": fname,
            "mse": mse,
            "n_kernels": n_kernels,
            "storage_torch_kb": storage_torch_kb,
            "storage_npz_kb": storage_npz_kb,
            **metrics,
        }

        # Save individual CSV
        individual_results_dir = os.path.join(
            config["output_dir"], "individual_results"
        )
        os.makedirs(individual_results_dir, exist_ok=True)
        individual_csv = os.path.join(individual_results_dir, f"{safe_name}.csv")
        pd.DataFrame([row]).to_csv(individual_csv, index=False)

        tune.report(row)

    except Exception as e:
        print(f"Error processing {fname}: {e}")

        # Save error CSV
        error_row = {"filename": fname, "error": str(e)}
        individual_results_dir = os.path.join(
            config["output_dir"], "individual_results"
        )
        os.makedirs(individual_results_dir, exist_ok=True)
        individual_csv = os.path.join(individual_results_dir, f"{safe_name}.csv")
        pd.DataFrame([error_row]).to_csv(individual_csv, index=False)

        tune.report({"mse": float("inf"), "error": str(e), "filename": fname})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel benchmark with Ray.")
    parser.add_argument("--root", type=str, default="/data2/objaverse")
    parser.add_argument(
        "--config", type=str, default="configs/uv_texture_fitting_knn.yaml"
    )
    parser.add_argument(
        "--rendering_config", type=str, default="configs/rendering.yaml"
    )
    parser.add_argument("--output_dir", type=str, default="outputs/benchmark")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
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

    args = parser.parse_args()

    print(f"Searching for .glb files in {args.root}...")
    all_filenames = find_filenames(args.root, file_ext=".glb")

    excluded_files = {  # files used for tuning
        "000-087/0e708d1e0ce0447ba5637a5320f5729c.glb",
        "000-018/998d641ce1c74e44978a91fedc849905.glb",
        "000-074/5ecf9d1175ae405a9a073db305786411.glb",
        "000-096/db5f9c28708142909b15212625a127f9.glb",
        "000-066/e7caba92073d4adba3477c21aa25e91f.glb",
    }
    all_filenames = [f for f in all_filenames if f not in excluded_files]

    all_filenames.sort()

    end_index = args.end_index if args.end_index != -1 else len(all_filenames)
    selected_filenames = all_filenames[args.start_index : end_index]

    print(f"Selected {len(selected_filenames)} files for benchmarking.")

    # Create individual results directory
    os.makedirs(os.path.join(args.output_dir, "individual_results"), exist_ok=True)

    search_space = {
        "filename": tune.grid_search(selected_filenames),
        "root": args.root,
        "base_config_path": os.path.abspath(args.config),
        "rendering_config_path": os.path.abspath(args.rendering_config),
        "output_dir": os.path.abspath(args.output_dir),
    }

    tuner = tune.Tuner(
        tune.with_resources(
            benchmark_trainable, resources={"gpu": args.gpus_per_trial}
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            max_concurrent_trials=args.max_concurrent_trials,
        ),
        run_config=tune.RunConfig(
            storage_path=os.path.abspath(args.output_dir),
            name="benchmark_run",
        ),
    )

    results = tuner.fit()
    df = results.get_dataframe()

    output_csv = os.path.join(args.output_dir, "benchmark_results.csv")
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            # Remove potential index columns from previous saves
            existing_df = existing_df.loc[
                :, ~existing_df.columns.str.contains("^Unnamed")
            ]
            df = pd.concat([existing_df, df], ignore_index=True)
            if "filename" in df.columns:
                df = df.drop_duplicates(subset=["filename"], keep="last")
        except Exception as e:
            print(f"Could not merge with existing results: {e}")

    df.to_csv(output_csv, index=False)
    print(f"Benchmark complete. Results saved to {args.output_dir}")
