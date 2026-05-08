import sys
import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import traceback

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
from ray import tune


def vertex_ray_benchmark_trainable(config):
    # Ensure imports in worker
    import mitsuba as mi
    import torch

    try:
        mi.set_variant("cuda_ad_rgb")
    except Exception:
        pass

    from heatsplats.trainers.vertex_ray_trainer import VertexRayTrainer
    from heatsplats.utils import mibitmaps2torch, compute_all_image_metrics
    from optimisation import main

    fname = config["filename"]
    target_kb = config["target_kb"]
    test_case = config["test_case"]
    root = config["root"]
    base_config_path = config["base_config_path"]
    rendering_config_path = config["rendering_config_path"]
    output_dir = config["output_dir"]

    if test_case == "gt":
        target_size_kb = None
    elif test_case == "hr_gt":
        target_size_kb = target_kb
    else:
        raise ValueError(f"Unknown test case {test_case}")

    args = argparse.Namespace(
        config=base_config_path,
        rendering_config=rendering_config_path,
        gpu="0",
        verbose=False,
    )

    trial_dir = tune.get_context().get_trial_dir()
    mesh_path = os.path.join(root, fname)

    # Handle filename for saving
    parts = fname.split("/")
    if "glbs" in parts:
        idx = parts.index("glbs")
        safe_name = "_".join(parts[idx + 1 :]).replace(".glb", "")
    else:
        safe_name = os.path.basename(fname).replace(".", "_")

    # Configure extras
    extras = [
        f"data.mesh_path={mesh_path}",
        f"exp_root_dir={trial_dir}",
        "name=output",
        f"tag={safe_name}",
        "use_timestamp=False",
        "optim.save_model=True",
        "optim.save_logs=True",
        f"trainer.target_size_kb={target_size_kb}",
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

        optimisation: VertexRayTrainer = out["optimisation"]
        mesh_vc = optimisation.mesh
        n_verts = mesh_vc.N_verts

        row = {
            "filename": fname,
            "target_kb": target_kb,
            "target_size_kb": target_size_kb,
            "test_case": test_case,
            "mse": mse,
            "storage_torch_kb": storage_torch_kb,
            "storage_npz_kb": storage_npz_kb,
            "n_verts": n_verts,
            **metrics,
        }

        # Save individual CSV
        individual_results_dir = os.path.join(
            output_dir, f"individual_results_vertex_ray_{test_case}"
        )
        os.makedirs(individual_results_dir, exist_ok=True)
        individual_csv = os.path.join(individual_results_dir, f"{safe_name}.csv")
        pd.DataFrame([row]).to_csv(individual_csv, index=False)

        tune.report(row)

    except Exception as e:
        print(f"Error processing {fname}: {e}")
        traceback.print_exc()
        tune.report({"mse": float("inf"), "error": str(e), "filename": fname})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vertex Ray benchmark with Ray.")
    parser.add_argument("--root", type=str, default="/data2/objaverse")
    parser.add_argument(
        "--benchmark_csv",
        type=str,
        default="outputs/benchmark_low_mem/benchmark_results.csv",
        help="Path to existing benchmark CSV with target sizes.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/vertex_colour_texture_fitting.yaml"
    )
    parser.add_argument(
        "--rendering_config", type=str, default="configs/rendering.yaml"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/vertex_ray_benchmark"
    )
    parser.add_argument(
        "--max_concurrent_trials",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--gpus_per_trial",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--test",
        nargs="+",
        default=["gt", "hr_gt"],
        help="List of test cases to benchmark",
    )

    args = parser.parse_args()

    # Load targets
    if not os.path.exists(args.benchmark_csv):
        print(f"Benchmark CSV not found: {args.benchmark_csv}")
        sys.exit(1)

    df_targets = pd.read_csv(args.benchmark_csv)
    # Filter out failed runs if any
    if "error" in df_targets.columns:
        df_targets = df_targets[
            df_targets["error"].isna() | (df_targets["error"] == "")
        ]

    # Prepare search space items
    items = []
    for _, row in df_targets.iterrows():
        items.append({"filename": row["filename"], "target_kb": row["storage_npz_kb"]})

    print(f"Found {len(items)} targets to process.")
    print(f"Running for encodings: {args.encodings}")

    # Run for each encoding separately to keep things organized
    for test in args.test:
        assert test in ["gt", "hr_gt"]
        print(f"\n{'='*80}\nRunning benchmark for test case: {test}\n{'='*80}")

        # Filter already processed
        output_csv = os.path.join(
            args.output_dir, f"benchmark_results_vertex_ray_{test}.csv"
        )
        current_items = items.copy()

        if os.path.exists(output_csv):
            try:
                existing_df = pd.read_csv(output_csv)
                if "filename" in existing_df.columns:
                    processed_files = set(existing_df["filename"])
                    current_items = [
                        i for i in current_items if i["filename"] not in processed_files
                    ]
                    print(
                        f"Skipping {len(items) - len(current_items)} already processed files."
                    )
            except Exception as e:
                print(f"Could not filter existing results: {e}")

        if not current_items:
            print(f"All files processed for {test}.")
            continue

        search_space = {
            "item": tune.grid_search(current_items),
            "test_case": test,
            "root": args.root,
            "base_config_path": os.path.abspath(args.config),
            "rendering_config_path": os.path.abspath(args.rendering_config),
            "output_dir": os.path.abspath(args.output_dir),
        }

        # Wrapper to unpack item
        def trainable_wrapper(config):
            item = config.pop("item")
            config["filename"] = item["filename"]
            config["target_kb"] = item["target_kb"]
            return vertex_ray_benchmark_trainable(config)

        tuner = tune.Tuner(
            tune.with_resources(
                trainable_wrapper, resources={"gpu": args.gpus_per_trial}
            ),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                max_concurrent_trials=args.max_concurrent_trials,
            ),
            run_config=tune.RunConfig(
                storage_path=os.path.abspath(args.output_dir),
                name=f"run_{test}",
            ),
        )

        results = tuner.fit()
        df_results = results.get_dataframe()

        # Merge and save
        if os.path.exists(output_csv):
            try:
                existing_df = pd.read_csv(output_csv)
                existing_df = existing_df.loc[
                    :, ~existing_df.columns.str.contains("^Unnamed")
                ]
                df_results = pd.concat([existing_df, df_results], ignore_index=True)
                if "filename" in df_results.columns:
                    df_results = df_results.drop_duplicates(
                        subset=["filename"], keep="last"
                    )
            except Exception as e:
                print(f"Could not merge with existing results: {e}")

        df_results.to_csv(output_csv, index=False)
        print(f"Results for {test} saved to {output_csv}")
