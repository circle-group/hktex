import argparse
import pandas as pd
import numpy as np
import os
import sys


def clean_filename(name):
    name = str(name)
    prefix = "hf-objaverse-v1/glbs/"
    return name[len(prefix) :] if name.startswith(prefix) else name


def main():
    parser = argparse.ArgumentParser(
        description="Calculate mean and std of benchmark metrics."
    )
    parser.add_argument("csv_file", type=str, help="Path to the CSV file.")
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

    if not os.path.exists(args.csv_file):
        print(f"File not found: {args.csv_file}")
        sys.exit(1)

    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Filter successful runs
    if "error" in df.columns:
        # Rows where 'error' is NaN or empty string are considered successful
        success_mask = df["error"].isna() | (df["error"] == "")
        successful_runs = df[success_mask]
    else:
        successful_runs = df

    if args.skip_file and os.path.exists(args.skip_file):
        with open(args.skip_file, "r") as f:
            skip_files = set(clean_filename(line.strip()) for line in f if line.strip())
        if "filename" in successful_runs.columns:
            initial_count = len(successful_runs)
            cleaned_filenames = successful_runs["filename"].apply(clean_filename)
            successful_runs = successful_runs[~cleaned_filenames.isin(skip_files)]
            skipped_count = initial_count - len(successful_runs)
            print(f"Filtered out {skipped_count} runs based on {args.skip_file}")

    if args.keep_file and os.path.exists(args.keep_file):
        with open(args.keep_file, "r") as f:
            keep_files = set(clean_filename(line.strip()) for line in f if line.strip())
        if "filename" in successful_runs.columns:
            initial_count = len(successful_runs)
            cleaned_filenames = successful_runs["filename"].apply(clean_filename)
            successful_runs = successful_runs[cleaned_filenames.isin(keep_files)]
            kept_count = len(successful_runs)
            print(
                f"Kept {kept_count} runs (filtered out {initial_count - kept_count}) based on {args.keep_file}"
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


if __name__ == "__main__":
    main()
