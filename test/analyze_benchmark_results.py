import argparse
import pandas as pd
import numpy as np
import os
import sys


def clean_filename(name):
    name = str(name)
    prefix = "hf-objaverse-v1/glbs/"
    return name[len(prefix) :] if name.startswith(prefix) else name


def load_csv(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error reading CSV file {path}: {e}")
        sys.exit(1)


def summarize_df(df, label=None, **kwargs):
    if label is not None:
        print(f"\n== {label} ==")

    successful_runs = get_successful_finite_runs(
        df,
        skip_file=kwargs.get("skip_file"),
        keep_file=kwargs.get("keep_file"),
    )

    num_successful = len(successful_runs)
    print(f"Total successful runs: {num_successful}")

    if num_successful == 0:
        return

    metrics = [
        "storage_npz_kb",
        "psnr",
        "ssim",
        "ms_ssim",
        "lpips",
        "n_kernels",
        "time_total_s",
        "mse",
        "n_verts",
        "n_fg_pixels",
        "nk-nv",
    ]

    print("-" * 85)
    print(f"{'Metric':<20} | {'Mean':<20} | {'Std':<20} | {'Clamped (MSE=0)':<20}")
    print("-" * 85)

    for metric in metrics:
        if metric in successful_runs.columns:
            data = pd.to_numeric(successful_runs[metric], errors="coerce")

            num_clamped = 0
            if metric == "psnr":
                inf_mask = np.isinf(data) & (data > 0)
                num_clamped = inf_mask.sum()
                if num_clamped > 0:
                    data[inf_mask] = 100.0

            # Replace infinity with NaN and drop them to avoid RuntimeWarnings and inf means
            data = data.replace([np.inf, -np.inf], np.nan).dropna()

            if not data.empty:
                mean_val = data.mean()
                std_val = data.std()
                if metric == "mse":
                    mean_val *= 1e3
                    std_val *= 1e3
                    metric = "mse (x1e-3)"
                clamped_str = str(num_clamped) if num_clamped > 0 else "-"
                print(
                    f"{metric:<20} | {mean_val:<20.6f} | {std_val:<20.6f} | {clamped_str:<20}"
                )
            else:
                print(f"{metric:<20} | {'N/A':<20} | {'N/A':<20} | {'-':<20}")
        else:
            print(f"{metric:<20} | {'N/A':<20} | {'N/A':<20} | {'-':<20}")
    print("-" * 85)


def get_successful_finite_runs(df, skip_file=None, keep_file=None):
    if "error" in df.columns:
        success_mask = df["error"].isna() | (df["error"] == "")
        successful_runs = df[success_mask]
    else:
        successful_runs = df

    mse_finite_mask = np.isfinite(successful_runs["mse"])
    if np.any(mse_finite_mask):
        successful_runs = successful_runs[mse_finite_mask]

    if skip_file is not None and os.path.exists(skip_file):
        with open(skip_file, "r") as f:
            skip_files = set(clean_filename(line.strip()) for line in f if line.strip())
        if "filename" in successful_runs.columns:
            initial_count = len(successful_runs)
            cleaned_filenames = successful_runs["filename"].apply(clean_filename)
            successful_runs = successful_runs[~cleaned_filenames.isin(skip_files)]
            skipped_count = initial_count - len(successful_runs)
            print(f"Filtered out {skipped_count} runs based on {skip_file}")

    if keep_file and os.path.exists(keep_file):
        with open(keep_file, "r") as f:
            keep_files = set(clean_filename(line.strip()) for line in f if line.strip())
        if "filename" in successful_runs.columns:
            initial_count = len(successful_runs)
            cleaned_filenames = successful_runs["filename"].apply(clean_filename)
            successful_runs = successful_runs[cleaned_filenames.isin(keep_files)]
            kept_count = len(successful_runs)
            print(
                f"Kept {kept_count} runs (filtered out {initial_count - kept_count}) based on {keep_file}"
            )

    return successful_runs


def summarize_csv_files(csv_files, **kwargs):
    dfs = [load_csv(path) for path in csv_files]
    summarize_df(pd.concat(dfs, ignore_index=True), **kwargs)


def collect_ablation_data(ablations_dirs, csv_name):
    grouped_ablation_data = {}
    num_roots = len(ablations_dirs)
    for ablations_dir in ablations_dirs:
        ablation_dirs = sorted(
            (path for path in os.scandir(ablations_dir) if path.is_dir()),
            key=lambda path: path.name,
        )
        if not ablation_dirs:
            print(f"No ablation directories found in: {ablations_dir}")
            sys.exit(1)

        for entry in ablation_dirs:
            csv_path = os.path.join(entry.path, csv_name)
            if not os.path.exists(csv_path):
                continue

            df = load_csv(csv_path)
            if "filename" not in df.columns:
                print(f"Missing 'filename' column in: {csv_path}")
                sys.exit(1)

            ablation_entry = grouped_ablation_data.setdefault(
                entry.name, {"csv_paths": [], "dfs": [], "roots": set()}
            )
            ablation_entry["csv_paths"].append(csv_path)
            ablation_entry["dfs"].append(df)
            ablation_entry["roots"].add(ablations_dir)

    complete_ablation_data = [
        (
            ablation_name,
            ablation_entry["csv_paths"],
            pd.concat(ablation_entry["dfs"], ignore_index=True),
        )
        for ablation_name, ablation_entry in sorted(grouped_ablation_data.items())
        if len(ablation_entry["roots"]) == num_roots
    ]
    skipped_ablation_names = [
        ablation_name
        for ablation_name, ablation_entry in sorted(grouped_ablation_data.items())
        if len(ablation_entry["roots"]) != num_roots
    ]

    return complete_ablation_data, skipped_ablation_names


def summarize_shared_ablation_results(ablations_dirs, csv_name, **kwargs):
    ablation_data, skipped_ablation_names = collect_ablation_data(
        ablations_dirs, csv_name
    )
    if skipped_ablation_names:
        print(
            "Skipping incomplete experiments missing from at least one ablation dir: "
            + ", ".join(skipped_ablation_names)
        )
    if not ablation_data:
        print("No complete experiments found across all provided ablation dirs.")
        sys.exit(1)

    shared_filenames = None
    for _, _, df in ablation_data:
        successful_runs = get_successful_finite_runs(df, **kwargs)
        filenames = set(successful_runs["filename"].dropna())
        shared_filenames = (
            filenames if shared_filenames is None else shared_filenames & filenames
        )

    print(
        f"Shared successful finite-MSE filenames across {len(ablation_data)} ablations "
        f"using {csv_name}: {len(shared_filenames)}"
    )

    for ablation_name, csv_paths, df in ablation_data:
        filtered_df = df[df["filename"].isin(shared_filenames)].copy()
        summarize_df(
            filtered_df, label=f"{ablation_name} ({', '.join(csv_paths)})", **kwargs
        )


def main():
    parser = argparse.ArgumentParser(
        description="Calculate mean and std of benchmark metrics."
    )
    parser.add_argument(
        "csv_files", nargs="*", type=str, help="Path(s) to CSV file(s)."
    )
    parser.add_argument(
        "--ablations-dir",
        nargs="+",
        type=str,
        help="One or more directories containing ablation subdirectories with benchmark CSVs.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="benchmark_results.csv",
        help="CSV filename to use inside each ablation directory (default: benchmark_results.csv).",
    )
    parser.add_argument(
        "--skip_file",
        type=str,
        default=None,
        help="Path to a txt file containing filenames to skip.",
    )
    parser.add_argument(
        "--keep_file",
        type=str,
        default=None,
        help="Path to a txt file containing filenames to exclusively keep.",
    )
    args = parser.parse_args()

    kwargs = {"skip_file": args.skip_file, "keep_file": args.keep_file}

    if args.ablations_dir:
        if args.csv_files:
            print("Do not combine positional csv_files with --ablations-dir.")
            sys.exit(1)
        summarize_shared_ablation_results(args.ablations_dir, args.csv_name, **kwargs)
        return

    if not args.csv_files:
        parser.error("Provide csv_files or use --ablations-dir.")

    summarize_csv_files(args.csv_files, **kwargs)


if __name__ == "__main__":
    main()
