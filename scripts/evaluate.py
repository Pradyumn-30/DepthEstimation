"""Evaluation script for depth estimation metrics.

Computes standard KITTI depth metrics on the Eigen test split:
- abs_rel: Mean absolute relative error
- sq_rel: Mean squared relative error
- rmse: Root mean squared error
- rmse_log: Root mean squared log error
- a1, a2, a3: Threshold accuracy (δ < 1.25, 1.25², 1.25³)

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/epoch_04.pt \\
                               --config configs/experiment_1.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from torch.utils.data import DataLoader

from depth.datasets.kitti_dataset import KITTIDataset
from depth.geometry.warping import disp_to_depth
from depth.models.depth_decoder import DepthDecoder
from depth.models.resnet_encoder import ResNetEncoder
from depth.utils.device import device_summary, get_device

# Depth clipping range for KITTI evaluation
MIN_DEPTH = 1e-3
MAX_DEPTH = 80.0


def compute_depth_metrics(
    pred: np.ndarray, gt: np.ndarray
) -> dict[str, float]:
    """Compute standard depth estimation metrics.

    Both pred and gt should be in meters, same spatial dimensions.
    Evaluation follows Eigen et al. protocol with Garg crop.

    Args:
        pred: Predicted depth map.
        gt: Ground truth depth map.

    Returns:
        Dict of metric names → values.
    """
    # Mask valid ground truth
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    # Clip to evaluation range
    pred = np.clip(pred, MIN_DEPTH, MAX_DEPTH)
    gt = np.clip(gt, MIN_DEPTH, MAX_DEPTH)

    # Median scaling (align predicted scale to ground truth)
    ratio = np.median(gt) / np.median(pred)
    pred *= ratio

    pred = np.clip(pred, MIN_DEPTH, MAX_DEPTH)

    # Compute metrics
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": a1,
        "a2": a2,
        "a3": a3,
    }


def evaluate(config: dict, checkpoint_path: str, split_name: str = "test"):
    """Run evaluation on specified split."""
    device = get_device(config.get("device", "auto"))
    print(f"Using device: {device_summary(device)}")

    data_cfg = config["data"]
    model_cfg = config["model"]

    # Load model
    encoder = ResNetEncoder(pretrained=False).to(device)
    decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["depth_encoder"])
    decoder.load_state_dict(checkpoint["depth_decoder"])
    encoder.eval()
    decoder.eval()

    # Load specified dataset split
    split_dir = Path("data/splits") / data_cfg["split"]
    split_file = split_dir / f"{split_name}_files.txt"
    
    if not split_file.exists():
        print(f"❌ Split file {split_file} not found.")
        return

    dataset = KITTIDataset(
        data_path=data_cfg["data_path"],
        split_path=str(split_file),
        height=data_cfg["height"],
        width=data_cfg["width"],
        frame_ids=[0],  # Only need target frame for eval
        is_train=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"\nEvaluating {len(dataset)} {split_name} images...")

    from depth.utils.kitti_utils import generate_depth_map
    import os

    all_metrics = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc=f"Evaluating ({split_name})")):
            image = batch[("color", 0)].to(device)

            # Predict disparity
            features = encoder(image)
            outputs = decoder(features)
            disp = outputs[("disp", 0)]  # (1, 1, H_net, W_net)

            # Load details for Velodyne projection
            date, drive, frame_idx, side = dataset.filenames[idx]
            
            # Construct path to Velodyne points .bin file
            velo_filename = os.path.join(
                data_cfg["data_path"], date, drive,
                "velodyne_points", "data", f"{frame_idx:010d}.bin"
            )
            calib_dir = os.path.join(data_cfg["data_path"], date)

            if not os.path.exists(velo_filename):
                continue

            # Project Velodyne points to sparse depth map
            cam = 2 if side == "l" else 3
            gt_depth = generate_depth_map(calib_dir, velo_filename, cam=cam, vel_depth=True)

            # Upsample predicted disparity to match original resolution
            disp_resized = F.interpolate(
                disp, size=gt_depth.shape,
                mode="bilinear", align_corners=False,
            )

            # Convert disparity to depth
            _, pred_depth = disp_to_depth(disp_resized)
            pred_depth = pred_depth.cpu().numpy()[0, 0]  # (H_orig, W_orig)

            # Apply Garg/Eigen crop
            h, w = gt_depth.shape
            eval_mask = np.zeros(gt_depth.shape, dtype=bool)
            y_min = int(0.40810811 * h)
            y_max = int(0.99189189 * h)
            x_min = int(0.03594771 * w)
            x_max = int(0.96405229 * w)
            eval_mask[y_min:y_max, x_min:x_max] = True

            # Filter valid gt pixels
            valid_gt = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
            mask = eval_mask & valid_gt

            if mask.sum() == 0:
                continue

            metrics = compute_depth_metrics(pred_depth[mask], gt_depth[mask])
            all_metrics.append(metrics)

    if all_metrics:
        # Average metrics
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS ({split_name.upper()} SPLIT)")
        print("=" * 60)
        print(f"  abs_rel: {avg['abs_rel']:.4f}")
        print(f"  sq_rel:  {avg['sq_rel']:.4f}")
        print(f"  rmse:    {avg['rmse']:.4f}")
        print(f"  rmse_log:{avg['rmse_log']:.4f}")
        print(f"  a1:      {avg['a1']:.4f}")
        print(f"  a2:      {avg['a2']:.4f}")
        print(f"  a3:      {avg['a3']:.4f}")
        print("=" * 60)
    else:
        print(f"\n❌ No valid ground truth depth maps could be processed for split: {split_name}.")
        print("Make sure Velodyne points are downloaded.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate depth model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Which split to evaluate: val or test",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluate(config, args.checkpoint, args.split)


if __name__ == "__main__":
    main()
