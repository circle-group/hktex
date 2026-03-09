import os
import argparse
import imageio
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from all MP4 videos in a directory."
    )
    parser.add_argument(
        "--root", type=str, default=".", help="Root directory to search for videos."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="extracted_frames",
        help="Directory to save extracted frames.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Find all mp4 files recursively
    videos = list(root_dir.rglob("*.mp4"))
    print(f"Found {len(videos)} videos in {root_dir}")

    for video_path in tqdm(videos, desc="Processing videos"):
        # Determine output path preserving directory structure relative to root
        try:
            rel_path = video_path.relative_to(root_dir)
            # Create a folder named after the video file (without extension)
            # inside the mirrored directory structure
            save_path = output_dir / rel_path.parent / rel_path.stem
        except ValueError:
            # Fallback if path manipulation fails
            save_path = output_dir / video_path.stem

        if save_path.exists() and any(save_path.iterdir()):
            # Skip if frames seem to already exist
            continue

        os.makedirs(save_path, exist_ok=True)

        try:
            reader = imageio.get_reader(str(video_path))
            for i, frame in enumerate(reader):
                Image.fromarray(frame).save(save_path / f"{i:05d}.png")
            reader.close()
        except Exception as e:
            print(f"Failed to extract {video_path}: {e}")


if __name__ == "__main__":
    main()
