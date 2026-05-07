import os
import shutil
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Move and rename benchmark videos.")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="outputs/benchmark_low_mem_extra/benchmark_run",
        help="Root directory to search for gt_vs_out.mp4 files.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="outputs/benchmark_low_mem_extra/benchmark_videos",
        help="Directory where videos will be moved.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="hf-objaverse-v1_glbs_",
        help="Prefix to strip from the folder name.",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print moves without executing them."
    )

    args = parser.parse_args()

    source = Path(args.source_dir)
    dest = Path(args.target_dir)

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {source}...")

    count = 0
    for root, dirs, files in os.walk(source):
        if "gt_vs_out.mp4" in files:
            file_path = Path(root) / "gt_vs_out.mp4"

            # Path structure: .../FOLDER_NAME/ckpts/gt_vs_out.mp4
            # We want FOLDER_NAME
            folder_name = file_path.parent.parent.name

            clean_name = folder_name.replace(args.prefix, "")
            new_filename = f"{clean_name}.mp4"
            dest_path = dest / new_filename

            print(f"Moving {file_path} -> {dest_path}")
            if not args.dry_run:
                shutil.move(str(file_path), str(dest_path))
            count += 1

    print(f"Finished. Moved {count} videos.")


if __name__ == "__main__":
    main()
