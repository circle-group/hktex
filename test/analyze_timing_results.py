import argparse
import pandas as pd
import numpy as np
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Calculate mean and std of benchmark metrics."
    )
    parser.add_argument("csv_file", type=str, help="Path to the CSV file.")
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

    num_successful = len(successful_runs)
    print(f"Total successful runs: {num_successful}")

    if num_successful == 0:
        return

    metrics = [
        "part1_renderer_init_s_mean",
        "part2_mesh_and_prepare_s_mean",
        "part3_render_s_mean",
        "total_s_mean",
        "time_total_s",
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
