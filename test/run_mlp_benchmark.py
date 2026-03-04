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


def calculate_hash_encoding_params(
    n_levels, log2_hashmap_size, n_features, base_res, scale
):
    total_params = 0
    max_params_per_level = 2**log2_hashmap_size

    for i in range(n_levels):
        resolution = int(np.ceil(base_res * (scale**i)))
        # Level size is min(dense_grid_size, hash_table_size)
        params_in_level = min(resolution**3, max_params_per_level)
        total_params += params_in_level * n_features
    return total_params


def find_best_hash_config(target_kb, input_dim=3, output_dim=3):
    # Fixed settings as per tinycudnn README example and common usage
    n_features = 2
    depth = 2  # Small MLP for hash encoding usually

    best_config = {
        "encoding.log2_hashmap_size": 19,
        "width": 64,
        "hidden": depth,
        "encoding.n_levels": 16,
        "encoding.n_features_per_level": n_features,
        "encoding.base_resolution": 16,
        "encoding.per_level_scale": 2.0,
        "encoding.interpolation": "Linear",
    }
    min_diff = float("inf")

    widths = [16, 32, 64]
    n_levels_options = [8, 12, 16, 20, 24]

    for width in widths:
        for n_levels in n_levels_options:
            # MLP size estimation (float32)
            # Hash encoding usually concatenates input
            mlp_in = n_levels * n_features + input_dim
            mlp_params = (
                (mlp_in * width + width)
                + (width * width + width) * (depth - 1)
                + (width * output_dim + output_dim)
            )
            mlp_size_kb = (mlp_params * 4) / 1024

            # Search log2_hashmap_size
            for log2 in range(6, 22):
                # Encoding params: n_levels * 2^log2 * n_features
                # TCNN uses float16 natively often, but MLPTextureNetwork requests float32
                enc_params = calculate_hash_encoding_params(
                    n_levels, log2, n_features, base_res=16, scale=2.0
                )
                total_kb = mlp_size_kb + (enc_params * 4) / 1024

                diff = abs(total_kb - target_kb)
                if total_kb > target_kb:
                    diff *= 1.5

                if diff < min_diff:
                    min_diff = diff
                    best_config["encoding.log2_hashmap_size"] = log2
                    best_config["width"] = width
                    best_config["encoding.n_levels"] = n_levels
                    best_config["estimated_kb"] = total_kb

    return best_config


def find_best_mlp_config(target_kb, input_dim, output_dim=3):
    best_config = {"width": 64, "hidden": 4}
    min_diff = float("inf")

    widths = [8, 16, 32, 64, 128, 256, 512]
    depths = [1, 2, 3, 4, 5, 6, 7, 8]

    for width in widths:
        for depth in depths:
            # Linear(in, width) + bias
            p = input_dim * width + width
            # Linear(width, width) + bias * (depth - 1)
            if depth > 1:
                p += (width * width + width) * (depth - 1)
            # Linear(width, out) + bias
            p += width * output_dim + output_dim

            size_kb = (p * 4) / 1024
            diff = abs(size_kb - target_kb)
            if size_kb > target_kb:
                diff *= 1.5

            if diff < min_diff:
                min_diff = diff
                best_config = {"width": width, "hidden": depth}

    return best_config


def mlp_benchmark_trainable(config):
    # Ensure imports in worker
    import mitsuba as mi
    import torch

    try:
        mi.set_variant("cuda_ad_rgb")
    except Exception:
        pass

    from heatsplats.utils import mibitmaps2torch, compute_all_image_metrics
    from optimisation import main

    fname = config["filename"]
    target_kb = config["target_kb"]
    encoding_type = config["encoding_type"]
    root = config["root"]
    base_config_path = config["base_config_path"]
    rendering_config_path = config["rendering_config_path"]
    output_dir = config["output_dir"]

    # Determine hyperparameters based on target size
    if encoding_type == "hash":
        # Input dim 3 for XYZ
        hparams = find_best_hash_config(target_kb, input_dim=3)
        print(
            f"File: {fname} | Target: {target_kb:.2f} KB | Est: {hparams.get('estimated_kb', 0):.2f} KB | Selected log2: {hparams['encoding.log2_hashmap_size']} | Levels: {hparams['encoding.n_levels']}"
        )
    elif encoding_type == "laplacian":
        # Input dim is k_eig (e.g., 32 or 64). Let's fix k_eig for embedding size
        # User said: "For laplacian encoding you do not need to take that into consideration as storage space"
        # We assume a fixed reasonable k_eig for the embedding input dimension
        k_eig = 64
        hparams = find_best_mlp_config(target_kb, input_dim=k_eig)
        hparams["encoding.eigen_albo_type"] = "modules.eigen-albo-interpolation"
        hparams["encoding.k_eig"] = 256  # Compute more, use subset
        hparams["encoding.effective_k_eig"] = k_eig  # Use subset for input
        # Need to serialize dict for OmegaConf if complex, but here simple keys work
    elif encoding_type == "positional":
        n_freqs = 6
        # Positional embedding output dim: 3 * 2 * n_ffreqs = 36
        input_dim = 3 * 2 * n_freqs
        hparams = find_best_mlp_config(target_kb, input_dim=input_dim)
        hparams["encoding.n_freqs"] = n_freqs
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

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
        f"trainer.network.encoding_type={encoding_type}",
        f"trainer.network.width={hparams['width']}",
        f"trainer.network.hidden={hparams['hidden']}",
    ]

    # Add encoding specific extras
    if encoding_type == "hash":
        extras.append("trainer.network.encoding.otype=HashGrid")
        extras.append("trainer.network.encoding.type=Hash")
        extras.append(
            f"trainer.network.encoding.n_levels={hparams['encoding.n_levels']}"
        )
        extras.append(
            f"trainer.network.encoding.n_features_per_level={hparams['encoding.n_features_per_level']}"
        )
        extras.append(
            f"trainer.network.encoding.log2_hashmap_size={hparams['encoding.log2_hashmap_size']}"
        )
        extras.append(
            f"trainer.network.encoding.base_resolution={hparams['encoding.base_resolution']}"
        )
        extras.append(
            f"trainer.network.encoding.per_level_scale={hparams['encoding.per_level_scale']}"
        )
        extras.append(
            f"trainer.network.encoding.interpolation={hparams['encoding.interpolation']}"
        )
    elif encoding_type == "laplacian":
        extras.append(f"trainer.network.encoding.k_eig={hparams['encoding.k_eig']}")
        extras.append(
            f"trainer.network.encoding.effective_k_eig={hparams['encoding.effective_k_eig']}"
        )
        extras.append("trainer.network.encoding.use_precomputed=True")
        extras.append(f"trainer.network.encoding.mesh_path={mesh_path}")
    elif encoding_type == "positional":
        extras.append(f"trainer.network.encoding.n_freqs={hparams['encoding.n_freqs']}")

    try:
        out = main(args, extras, render=True)

        gt_rend, result_rend, _, _ = out["renderings"]
        gt = mibitmaps2torch(gt_rend)
        res = mibitmaps2torch(result_rend)

        mse = F.mse_loss(res, gt, reduction="mean").item()
        metrics = compute_all_image_metrics(res, gt)

        storage_torch_kb = out["storage"][0]
        storage_npz_kb = out["storage"][1]

        row = {
            "filename": fname,
            "target_kb": target_kb,
            "encoding": encoding_type,
            "mse": mse,
            "storage_torch_kb": storage_torch_kb,
            "storage_npz_kb": storage_npz_kb,
            "param_width": hparams["width"],
            "param_hidden": hparams["hidden"],
            **metrics,
        }

        if encoding_type == "hash":
            row["param_log2_hashmap"] = hparams.get("encoding.log2_hashmap_size")

        # Save individual CSV
        individual_results_dir = os.path.join(
            output_dir, f"individual_results_mlp_{encoding_type}"
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
    parser = argparse.ArgumentParser(description="Run MLP benchmark with Ray.")
    parser.add_argument("--root", type=str, default="/data2/objaverse")
    parser.add_argument(
        "--benchmark_csv",
        type=str,
        default="outputs/benchmark_low_mem/benchmark_results.csv",
        help="Path to existing benchmark CSV with target sizes.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/uv_texture_mlp_fitting.yaml"
    )
    parser.add_argument(
        "--rendering_config", type=str, default="configs/rendering.yaml"
    )
    parser.add_argument("--output_dir", type=str, default="outputs/mlp_benchmark")
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
        "--encodings",
        nargs="+",
        default=["hash", "laplacian", "positional"],
        help="List of encodings to benchmark",
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

    # Run for each encoding separately to keep things organized
    for encoding in args.encodings:
        print(f"\n{'='*80}\nRunning benchmark for encoding: {encoding}\n{'='*80}")

        # Filter already processed
        output_csv = os.path.join(
            args.output_dir, f"benchmark_results_mlp_{encoding}.csv"
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
            print(f"All files processed for {encoding}.")
            continue

        search_space = {
            "item": tune.grid_search(current_items),
            "encoding_type": encoding,
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
            return mlp_benchmark_trainable(config)

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
                name=f"run_{encoding}",
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
        print(f"Results for {encoding} saved to {output_csv}")
