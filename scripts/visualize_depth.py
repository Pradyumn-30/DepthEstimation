"""Depth map visualization script.

Loads a trained model checkpoint and generates side-by-side visual comparisons
between input RGB images and predicted depth maps colorized with a heatmap.

Usage:
    # 1. Visualize a single image:
    python scripts/visualize_depth.py --checkpoint checkpoints/epoch_04.pt \
                                      --config configs/local_debug.yaml \
                                      --image data/kitti_raw/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png \
                                      --output outputs/depth_vis.png

    # 2. Visualize all frames in a drive sequence:
    python scripts/visualize_depth.py --checkpoint checkpoints/epoch_04.pt \
                                      --config configs/local_debug.yaml \
                                      --drive data/kitti_raw/2011_09_26/2011_09_26_drive_0001_sync \
                                      --output outputs/sequence_vis
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from depth.geometry.warping import disp_to_depth
from depth.models.depth_decoder import DepthDecoder
from depth.models.resnet_encoder import ResNetEncoder
from depth.utils.device import get_device


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load depth network from checkpoint."""
    encoder = ResNetEncoder(pretrained=False).to(device)
    decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["depth_encoder"])
    decoder.load_state_dict(checkpoint["depth_decoder"])

    encoder.eval()
    decoder.eval()
    return encoder, decoder


def predict_depth(
    image_path: str,
    encoder: ResNetEncoder,
    decoder: DepthDecoder,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Load image, run forward pass, return original image and depth array."""
    # Load and resize original PIL Image
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    # Resize image to network input resolution
    input_img = pil_img.resize((width, height), Image.LANCZOS)
    input_tensor = TF.to_tensor(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(input_tensor)
        outputs = decoder(features)
        disp = outputs[("disp", 0)]  # (1, 1, height, width)

        # Upsample predicted disparity back to original image size
        disp_resized = F.interpolate(
            disp, size=(orig_h, orig_w),
            mode="bilinear", align_corners=False,
        )

        _, depth_tensor = disp_to_depth(disp_resized)
        depth = depth_tensor.squeeze().cpu().numpy()

    return np.array(pil_img), depth


def save_visualization(
    rgb_img: np.ndarray,
    depth_map: np.ndarray,
    output_path: str,
    colormap: str = "magma",
):
    """Save side-by-side RGB and Depth Heatmap comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # RGB image
    axes[0].imshow(rgb_img)
    axes[0].set_title("Input RGB Image")
    axes[0].axis("off")

    # Depth Heatmap
    # Normalize depth map for better visualization contrast
    # Use log depth for better visual gradient spacing
    log_depth = np.log(depth_map + 1e-3)
    im = axes[1].imshow(log_depth, cmap=colormap)
    axes[1].set_title("Predicted Depth Map (Heatmap)")
    axes[1].axis("off")

    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Relative Log Depth")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize depth predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--image", type=str, default="", help="Path to single image")
    parser.add_argument("--drive", type=str, default="", help="Path to drive folder")
    parser.add_argument("--output", type=str, default="outputs/vis.png", help="Output path/directory")
    parser.add_argument("--colormap", type=str, default="magma", help="Matplotlib colormap (magma/inferno/plasma)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = get_device(config.get("device", "auto"))
    encoder, decoder = load_model(args.checkpoint, config, device)

    height = config["data"]["height"]
    width = config["data"]["width"]

    # 1. Single Image Mode
    if args.image:
        rgb_img, depth = predict_depth(args.image, encoder, decoder, height, width, device)
        save_visualization(rgb_img, depth, args.output, args.colormap)

    # 2. Drive Sequence Mode
    elif args.drive:
        img_dir = Path(args.drive) / "image_02" / "data"
        if not img_dir.exists():
            print(f"❌ image_02 directory not found under {args.drive}")
            sys.exit(1)

        images = sorted(list(img_dir.glob("*.png")))
        if not images:
            print(f"No images found in {img_dir}")
            sys.exit(1)

        print(f"Processing sequence: {len(images)} frames found")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process a subset (first 20 frames) to keep visualization fast
        for i, img_path in enumerate(images[:20]):
            rgb_img, depth = predict_depth(img_path, encoder, decoder, height, width, device)
            frame_output = output_dir / f"frame_{i:04d}.png"
            save_visualization(rgb_img, depth, str(frame_output), args.colormap)

    else:
        print("❌ Please specify either --image or --drive parameter to visualize.")


if __name__ == "__main__":
    main()
