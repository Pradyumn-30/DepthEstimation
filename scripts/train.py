"""Training script for self-supervised monocular depth estimation.

Usage:
    python scripts/train.py --config configs/experiment_1.yaml
    python scripts/train.py --config configs/default.yaml  # Phase 2 (AWS)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from depth.datasets.kitti_dataset import KITTIDataset
from depth.geometry.projection import BackprojectDepth, Project3D
from depth.geometry.warping import (
    disp_to_depth,
    transformation_from_parameters,
    warp_image,
)
from depth.losses.masking import apply_auto_mask, compute_min_reprojection
from depth.losses.photometric import SSIM, compute_reprojection_loss
from depth.losses.smoothness import compute_smoothness_loss
from depth.models.depth_decoder import DepthDecoder
from depth.models.pose_decoder import PoseDecoder
from depth.models.resnet_encoder import ResNetEncoder
from depth.utils.device import device_summary, get_device


class Trainer:
    """Self-supervised monocular depth trainer.

    Manages the full training loop: model creation, loss computation,
    optimization, logging, and checkpointing.
    """

    def __init__(self, config: dict, resume_path: str = None):
        self.config = config
        self.device = get_device(config.get("device", "auto"))
        print(f"Using device: {device_summary(self.device)}")

        # Training params
        train_cfg = config["training"]
        self.batch_size = train_cfg["batch_size"]
        self.num_epochs = train_cfg["num_epochs"]
        self.learning_rate = train_cfg["learning_rate"]
        self.log_freq = train_cfg.get("log_frequency", 50)
        self.save_freq = train_cfg.get("save_frequency", 1)

        # Model params
        model_cfg = config["model"]
        self.frame_ids = model_cfg["frame_ids"]
        self.num_scales = model_cfg["num_scales"]
        self.scales = list(range(self.num_scales))

        # Loss params
        loss_cfg = config["loss"]
        self.ssim_weight = loss_cfg["ssim_weight"]
        self.smoothness_weight = loss_cfg["smoothness_weight"]
        self.use_auto_mask = loss_cfg["auto_mask"]
        self.use_min_reproj = loss_cfg["min_reprojection"]

        # Data params
        data_cfg = config["data"]
        self.height = data_cfg["height"]
        self.width = data_cfg["width"]

        # Create output directories
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path("runs") / time.strftime("%Y%m%d_%H%M%S")

        self._build_models()
        self._build_dataset(data_cfg)
        self._build_optimizer(train_cfg)
        self._build_geometry()

        # Loss modules
        self.ssim = SSIM().to(self.device)

        # TensorBoard
        self.writer = SummaryWriter(str(self.log_dir))
        self.global_step = 0
        self.start_epoch = 0

        # Load checkpoint if requested
        if resume_path:
            self._load_checkpoint(resume_path)

    def _build_models(self):
        """Initialize depth encoder, depth decoder, pose encoder, pose decoder."""
        model_cfg = self.config["model"]
        pretrained = model_cfg.get("pretrained", True)

        # Depth network
        self.depth_encoder = ResNetEncoder(
            pretrained=pretrained, num_input_images=1
        ).to(self.device)

        self.depth_decoder = DepthDecoder(
            num_ch_enc=self.depth_encoder.num_ch_enc,
            scales=self.scales,
        ).to(self.device)

        # Pose network (separate encoder for 2-image input)
        self.pose_encoder = ResNetEncoder(
            pretrained=pretrained, num_input_images=2
        ).to(self.device)

        self.pose_decoder = PoseDecoder(
            num_ch_enc=self.pose_encoder.num_ch_enc,
            num_frames_to_predict_for=2,
        ).to(self.device)

        # Collect all parameters
        self.parameters_to_train = (
            list(self.depth_encoder.parameters())
            + list(self.depth_decoder.parameters())
            + list(self.pose_encoder.parameters())
            + list(self.pose_decoder.parameters())
        )

        total_params = sum(p.numel() for p in self.parameters_to_train)
        print(f"Total trainable parameters: {total_params:,}")

    def _build_dataset(self, data_cfg: dict):
        """Create train and val dataloaders."""
        split_dir = Path("data/splits") / data_cfg["split"]
        aug_cfg = self.config.get("augmentation", {})
        color_jitter = aug_cfg.get("color_jitter", {})
        color_jitter["horizontal_flip"] = aug_cfg.get("horizontal_flip", 0.5)

        self.train_dataset = KITTIDataset(
            data_path=data_cfg["data_path"],
            split_path=str(split_dir / "train_files.txt"),
            height=data_cfg["height"],
            width=data_cfg["width"],
            frame_ids=self.frame_ids,
            is_train=True,
            augmentation_config=color_jitter,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=data_cfg.get("num_workers", 2),
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )

        # Val loader (optional, may not have val split populated yet)
        val_split = split_dir / "val_files.txt"
        if val_split.exists():
            self.val_dataset = KITTIDataset(
                data_path=data_cfg["data_path"],
                split_path=str(val_split),
                height=data_cfg["height"],
                width=data_cfg["width"],
                frame_ids=self.frame_ids,
                is_train=False,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=data_cfg.get("num_workers", 2),
                drop_last=False,
            )
        else:
            self.val_loader = None

    def _build_optimizer(self, train_cfg: dict):
        """Create Adam optimizer and step LR scheduler."""
        self.optimizer = torch.optim.Adam(
            self.parameters_to_train,
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=train_cfg.get("scheduler_step", 15),
            gamma=train_cfg.get("scheduler_gamma", 0.1),
        )

    def _build_geometry(self):
        """Pre-allocate geometry modules for each scale."""
        self.backproject = {}
        self.project = {}

        for scale in self.scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)

            self.backproject[scale] = BackprojectDepth(
                self.batch_size, h, w
            ).to(self.device)

            self.project[scale] = Project3D(
                self.batch_size, h, w
            ).to(self.device)

    def train(self):
        """Run the full training loop."""
        print(f"\nStarting training: {self.num_epochs} epochs, "
              f"{len(self.train_loader)} batches/epoch")
        print(f"Resolution: {self.height}x{self.width}, "
              f"Batch size: {self.batch_size}")
        print(f"Logging to: {self.log_dir}\n")

        for epoch in range(self.start_epoch, self.num_epochs):
            self._train_epoch(epoch)
            self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.save_freq == 0:
                self._save_checkpoint(epoch)

            # Clear MPS cache if applicable
            if self.device.type == "mps":
                torch.mps.empty_cache()

        self.writer.close()
        print("\nTraining complete!")

    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.depth_encoder.train()
        self.depth_decoder.train()
        self.pose_encoder.train()
        self.pose_decoder.train()

        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            # Move data to device
            inputs = self._move_to_device(batch)

            # Forward pass: depth
            features = self.depth_encoder(inputs[("color_aug", 0)])
            depth_outputs = self.depth_decoder(features)

            # Forward pass: pose (for each source frame pair)
            pose_outputs = self._predict_poses(inputs)

            # Compute losses
            losses = self._compute_losses(inputs, depth_outputs, pose_outputs)
            total_loss = losses["loss"]

            # Backward + optimize
            total_loss.backward()
            self.optimizer.step()

            # Logging
            epoch_loss += total_loss.item()
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

            if self.global_step % self.log_freq == 0:
                self._log_step(losses, depth_outputs, inputs)

            self.global_step += 1

        avg_loss = epoch_loss / len(self.train_loader)
        print(f"  Epoch {epoch+1} average loss: {avg_loss:.4f}")
        self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)

    def _move_to_device(self, batch: dict) -> dict:
        """Move all tensors in batch to device."""
        inputs = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                inputs[key] = val.to(self.device)
            else:
                inputs[key] = val
        return inputs

    def _predict_poses(self, inputs: dict) -> dict:
        """Predict relative poses for each source frame.

        The pose network takes pairs of frames and predicts relative
        6-DOF transforms. For frame_ids [0, -1, 1]:
        - Input pair (I_{-1}, I_0) → pose from -1 to 0
        - Input pair (I_1, I_0) → pose from 1 to 0

        Returns:
            Dict mapping ("axisangle", source_fid, target_fid) and
            ("translation", source_fid, target_fid) to tensors,
            plus ("cam_T_cam", source_fid, target_fid) for the full 4x4 matrix.
        """
        outputs = {}

        # Source frame IDs (exclude target frame 0)
        source_ids = [fid for fid in self.frame_ids if fid != 0]

        # Stack source frames with target for pose prediction
        # Pose encoder expects (B, 6, H, W) = concat of 2 RGB images
        for i, fid in enumerate(source_ids):
            # Order: (source, target) if fid < 0, (target, source) if fid > 0
            if fid < 0:
                pose_input = torch.cat([
                    inputs[("color_aug", fid)],
                    inputs[("color_aug", 0)]
                ], dim=1)
            else:
                pose_input = torch.cat([
                    inputs[("color_aug", 0)],
                    inputs[("color_aug", fid)]
                ], dim=1)

            features = self.pose_encoder(pose_input)
            pose_pred = self.pose_decoder(features)  # (B, 2, 6)

            # We predict poses for both orderings but only need one
            # pose_pred[:, 0, :] corresponds to the first pair in the batch
            axisangle = pose_pred[:, 0, :3].unsqueeze(1)  # (B, 1, 3)
            translation = pose_pred[:, 0, 3:].unsqueeze(1)  # (B, 1, 3)

            outputs[("axisangle", fid)] = axisangle
            outputs[("translation", fid)] = translation

            # Build 4x4 transformation matrix
            T = transformation_from_parameters(
                axisangle, translation, invert=(fid < 0)
            )
            outputs[("cam_T_cam", fid)] = T

        return outputs

    def _compute_losses(
        self, inputs: dict, depth_outputs: dict, pose_outputs: dict
    ) -> dict:
        """Compute all losses across scales.

        For each scale:
        1. Convert disparity → depth
        2. Warp each source frame → synthesized target
        3. Compute photometric loss (L1 + SSIM)
        4. Apply min-reprojection across source frames
        5. Apply auto-masking
        6. Add smoothness loss

        Returns:
            Dict with "loss" (total scalar) and component losses.
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)

        source_ids = [fid for fid in self.frame_ids if fid != 0]
        target = inputs[("color", 0)]  # Clean target for loss

        for scale in self.scales:
            # Get disparity at this scale
            disp = depth_outputs[("disp", scale)]

            # Upsample disparity to full resolution for warping
            if scale > 0:
                disp = F.interpolate(
                    disp, size=(self.height, self.width),
                    mode="bilinear", align_corners=False,
                )

            _, depth = disp_to_depth(disp)

            # Warp each source frame
            reprojection_losses = []
            for fid in source_ids:
                source = inputs[("color", fid)]
                T = pose_outputs[("cam_T_cam", fid)]

                warped = warp_image(
                    source, depth, T,
                    inputs["K"], inputs["inv_K"],
                    self.backproject[0],  # Always warp at full res
                    self.project[0],
                )

                reproj_loss = compute_reprojection_loss(
                    warped, target, self.ssim, self.ssim_weight
                )
                reprojection_losses.append(reproj_loss)

            # Min reprojection across source frames
            if self.use_min_reproj:
                reproj_loss = compute_min_reprojection(reprojection_losses)
            else:
                reproj_loss = sum(reprojection_losses) / len(reprojection_losses)

            # Auto-masking
            if self.use_auto_mask:
                identity_losses = []
                for fid in source_ids:
                    identity_loss = compute_reprojection_loss(
                        inputs[("color", fid)], target,
                        self.ssim, self.ssim_weight,
                    )
                    identity_losses.append(identity_loss)

                reproj_loss = apply_auto_mask(reproj_loss, identity_losses)

            losses[f"reproj_loss/{scale}"] = reproj_loss.mean()

            # Smoothness loss
            # Use the disparity at native scale (not upsampled)
            native_disp = depth_outputs[("disp", scale)]
            # Downsample target image to match disparity scale
            if scale > 0:
                target_scaled = F.interpolate(
                    target, size=native_disp.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            else:
                target_scaled = target

            smooth_loss = compute_smoothness_loss(native_disp, target_scaled)
            smooth_loss = self.smoothness_weight * smooth_loss / (2 ** scale)
            losses[f"smooth_loss/{scale}"] = smooth_loss

            total_loss += reproj_loss.mean() + smooth_loss

        # Average across scales
        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def _log_step(self, losses: dict, depth_outputs: dict, inputs: dict):
        """Log losses and visualizations to TensorBoard."""
        for key, val in losses.items():
            self.writer.add_scalar(f"train/{key}", val.item(), self.global_step)

        # Log depth visualization (scale 0 only, first item in batch)
        if self.global_step % (self.log_freq * 5) == 0:
            disp = depth_outputs[("disp", 0)][0].detach()
            # Normalize to [0, 1] for visualization
            disp_vis = (disp - disp.min()) / (disp.max() - disp.min() + 1e-7)
            self.writer.add_image(
                "train/disp_0", disp_vis, self.global_step
            )
            self.writer.add_image(
                "train/input", inputs[("color", 0)][0].detach().cpu(),
                self.global_step,
            )

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "depth_encoder": self.depth_encoder.state_dict(),
            "depth_decoder": self.depth_decoder.state_dict(),
            "pose_encoder": self.pose_encoder.state_dict(),
            "pose_decoder": self.pose_decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }
        path = self.checkpoint_dir / f"epoch_{epoch:02d}.pt"
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str):
        """Load model weights and optimizer/scheduler state from checkpoint."""
        print(f"Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.depth_encoder.load_state_dict(checkpoint["depth_encoder"])
        self.depth_decoder.load_state_dict(checkpoint["depth_decoder"])
        self.pose_encoder.load_state_dict(checkpoint["pose_encoder"])
        self.pose_decoder.load_state_dict(checkpoint["pose_decoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        print(f"  Resumed at epoch {self.start_epoch}, global step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description="Train self-supervised depth model")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Self-Supervised Monocular Depth Estimation")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"PyTorch version: {torch.__version__}")

    trainer = Trainer(config, args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
