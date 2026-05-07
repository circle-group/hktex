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
        description="Find common valid filenames across multiple benchmark CSV files."
    )
    parser.add_argument(
        "csv_files", nargs="+", type=str, help="Paths to the CSV files to compare."
    )
    parser.add_argument(
        "--skip_file",
        type=str,
        default=None,
        help="Path to a txt file containing filenames to skip.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="common_valid_files.txt",
        help="Path to output the resulting text file.",
    )
    args = parser.parse_args()

    skip_files = set()
    if args.skip_file and os.path.exists(args.skip_file):
        with open(args.skip_file, "r") as f:
            skip_files = set(clean_filename(line.strip()) for line in f if line.strip())
        print(f"Loaded {len(skip_files)} files to skip.")

    common_filenames = None

    for csv_file in args.csv_files:
        if not os.path.exists(csv_file):
            print(f"Warning: File not found: {csv_file}")
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        if "filename" not in df.columns:
            print(f"Warning: 'filename' column missing in {csv_file}, skipping.")
            continue

        # Filter successful runs
        if "error" in df.columns:
            success_mask = df["error"].isna() | (df["error"] == "")
            df = df[success_mask]

        valid_filenames = set(df["filename"].dropna().apply(clean_filename))
        valid_filenames = valid_filenames - skip_files

        if common_filenames is None:
            common_filenames = valid_filenames
        else:
            common_filenames = common_filenames.intersection(valid_filenames)

    if common_filenames is None or len(common_filenames) == 0:
        print("No common valid files found across the provided CSVs.")
        sys.exit(1)

    print(
        f"Found {len(common_filenames)} common valid filenames. Saving to {args.output}..."
    )
    with open(args.output, "w") as f:
        for fname in sorted(common_filenames):
            f.write(f"{fname}\n")


if __name__ == "__main__":
    main()
