"""Script to compile generated PNG frames into an animated GIF.

Usage:
    python scripts/compile_gif.py --input outputs/sequence_vis_cars \
                                  --output outputs/depth_animation.gif \
                                  --fps 5
"""

import argparse
import sys
from pathlib import Path
from PIL import Image


def compile_gif(input_dir: str, output_path: str, fps: int):
    """Load all PNG files in directory and compile into animated GIF."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory {input_dir} does not exist.")
        sys.exit(1)

    # Find and sort all PNG frames
    frames = sorted(list(input_path.glob("frame_*.png")))
    if not frames:
        print(f"❌ No frame_*.png files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(frames)} frames. Compiling...")

    # Load images
    images = [Image.open(str(f)) for f in frames]

    # Calculate duration per frame in milliseconds
    duration = int(1000 / fps)

    # Save as animated GIF
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    images[0].save(
        str(output_file),
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,  # Loop forever
    )

    print(f"✅ Animated GIF successfully saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compile PNG frames into GIF")
    parser.add_argument("--input", type=str, required=True, help="Directory containing PNG frames")
    parser.add_argument("--output", type=str, default="outputs/depth_animation.gif", help="Output GIF path")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    args = parser.parse_args()

    compile_gif(args.input, args.output, args.fps)


if __name__ == "__main__":
    main()
