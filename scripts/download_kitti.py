"""Download KITTI raw data for the curated Eigen split.

Downloads only the specific drives needed for the curated subset,
then generates the full split files (enumerating all frame indices).

Usage:
    python scripts/download_kitti.py --output data/kitti_raw --split eigen_curated
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Drives in the curated Eigen split
CURATED_DRIVES = {
    "train": [
        ("2011_09_26", "2011_09_26_drive_0001"),
        ("2011_09_26", "2011_09_26_drive_0014"),
        ("2011_09_26", "2011_09_26_drive_0020"),
        ("2011_09_26", "2011_09_26_drive_0056"),
        ("2011_09_26", "2011_09_26_drive_0059"),
        ("2011_09_26", "2011_09_26_drive_0084"),
        ("2011_09_26", "2011_09_26_drive_0091"),
        ("2011_09_28", "2011_09_28_drive_0001"),
    ],
    "val": [
        ("2011_09_29", "2011_09_29_drive_0004"),
    ],
    "test": [
        ("2011_09_30", "2011_09_30_drive_0016"),
    ],
}

# KITTI raw data base URL
KITTI_BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"


def download_drive(date: str, drive: str, output_dir: Path):
    """Download a single KITTI drive (synced+rectified)."""
    drive_sync = f"{drive}_sync"
    zip_name = f"{drive_sync}.zip"
    url = f"{KITTI_BASE_URL}/{drive}/{zip_name}"

    date_dir = output_dir / date
    drive_dir = date_dir / drive_sync

    if drive_dir.exists() and any(drive_dir.rglob("*.png")):
        print(f"  ✓ {drive_sync} already exists, skipping")
        return

    date_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {zip_name}...")
    zip_path = date_dir / zip_name

    try:
        subprocess.run(
            ["curl", "-L", "-o", str(zip_path), url],
            check=True,
            capture_output=True,
        )

        print(f"  Extracting {zip_name}...")
        subprocess.run(
            ["unzip", "-q", "-o", str(zip_path), "-d", str(output_dir)],
            check=True,
            capture_output=True,
        )

        # Clean up zip
        zip_path.unlink()
        print(f"  ✓ {drive_sync} ready")

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to download {drive_sync}: {e}")
        if zip_path.exists():
            zip_path.unlink()


def download_calibration(date: str, output_dir: Path):
    """Download calibration files for a given date."""
    calib_file = output_dir / date / "calib_cam_to_cam.txt"
    if calib_file.exists():
        print(f"  ✓ Calibration for {date} already exists")
        return

    url = f"{KITTI_BASE_URL}/{date}_calib.zip"
    zip_path = output_dir / f"{date}_calib.zip"

    print(f"  Downloading calibration for {date}...")
    try:
        subprocess.run(
            ["curl", "-L", "-o", str(zip_path), url],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["unzip", "-q", "-o", str(zip_path), "-d", str(output_dir)],
            check=True,
            capture_output=True,
        )
        zip_path.unlink()
        print(f"  ✓ Calibration for {date} ready")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {e}")
        if zip_path.exists():
            zip_path.unlink()


def generate_split_files(output_dir: Path, split_output_dir: Path):
    """Generate split files by enumerating all available frames.

    Scans downloaded drives and creates train/val/test_files.txt with
    one entry per valid frame (excluding first and last frame in each
    drive, since they lack temporal neighbors).
    """
    split_output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, drives in CURATED_DRIVES.items():
        lines = []
        for date, drive in drives:
            drive_sync = f"{drive}_sync"
            img_dir = output_dir / date / drive_sync / "image_02" / "data"

            if not img_dir.exists():
                print(f"  Warning: {img_dir} not found, skipping")
                continue

            # Count frames
            frames = sorted(img_dir.glob("*.png"))
            num_frames = len(frames)

            if num_frames < 3:
                print(f"  Warning: {drive_sync} has only {num_frames} frames, skipping")
                continue

            # Exclude first and last frame (no temporal neighbor)
            for idx in range(1, num_frames - 1):
                lines.append(f"{date} {drive_sync} {idx} l")

        # Write split file
        split_file = split_output_dir / f"{split_name}_files.txt"
        with open(split_file, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"  {split_name}: {len(lines)} frames → {split_file}")


def main():
    parser = argparse.ArgumentParser(description="Download KITTI raw data")
    parser.add_argument(
        "--output", type=str, default="data/kitti_raw",
        help="Output directory for raw data",
    )
    parser.add_argument(
        "--split", type=str, default="eigen_curated",
        help="Split name (eigen_curated or eigen)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, only regenerate split files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    split_dir = Path("data/splits") / args.split

    # Ensure output_dir exists immediately
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KITTI Raw Data Download")
    print("=" * 60)

    if not args.skip_download:
        # Collect all unique dates for calibration download
        all_dates = set()
        all_drives = []
        for split_drives in CURATED_DRIVES.values():
            for date, drive in split_drives:
                all_dates.add(date)
                all_drives.append((date, drive))

        # Download calibration files
        print("\nDownloading calibration files...")
        for date in sorted(all_dates):
            # Ensure date folder exists for calibration extraction
            (output_dir / date).mkdir(parents=True, exist_ok=True)
            download_calibration(date, output_dir)

        # Download drives
        print(f"\nDownloading {len(all_drives)} drives...")
        for date, drive in all_drives:
            download_drive(date, drive, output_dir)

    # Generate split files
    print("\nGenerating split files...")
    generate_split_files(output_dir, split_dir)

    print("\nDone! You can now train with:")
    print(f"  python scripts/train.py --config configs/experiment_1.yaml")


if __name__ == "__main__":
    main()
