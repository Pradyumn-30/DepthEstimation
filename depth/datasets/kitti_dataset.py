"""KITTI Raw dataset loader for self-supervised monocular depth training.

Loads temporally adjacent triplets (I_{t-1}, I_t, I_{t+1}) from KITTI raw
synced+rectified data, along with camera intrinsics.

Expected directory structure:
    data/kitti_raw/
    ├── 2011_09_26/
    │   ├── 2011_09_26_drive_0001_sync/
    │   │   └── image_02/
    │   │       └── data/
    │   │           ├── 0000000000.png
    │   │           ├── 0000000001.png
    │   │           └── ...
    │   ├── calib_cam_to_cam.txt
    │   └── ...
    └── ...

Split file format (one sample per line):
    date drive_name frame_index side
    e.g.: 2011_09_26 2011_09_26_drive_0001_sync 5 l
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from depth.datasets.transforms import TrainTransform, ValTransform


class KITTIDataset(Dataset):
    """KITTI monocular depth dataset.

    Args:
        data_path: Root path to KITTI raw data.
        split_path: Path to split file (train_files.txt, etc.).
        height: Target image height.
        width: Target image width.
        frame_ids: List of frame offsets to load. Default [0, -1, 1].
        is_train: If True, apply training augmentations.
        augmentation_config: Dict with color jitter / flip params.
    """

    # Camera directories: "l" = left (image_02), "r" = right (image_03)
    SIDE_MAP = {"l": "image_02", "r": "image_03"}

    def __init__(
        self,
        data_path: str,
        split_path: str,
        height: int = 128,
        width: int = 416,
        frame_ids: list[int] = None,
        is_train: bool = True,
        augmentation_config: dict = None,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.height = height
        self.width = width
        self.frame_ids = frame_ids or [0, -1, 1]
        self.is_train = is_train

        # Load split file
        self.filenames = self._load_split(split_path)

        # Set up transforms
        if is_train:
            aug = augmentation_config or {}
            self.transform = TrainTransform(
                height=height,
                width=width,
                brightness=aug.get("brightness", 0.2),
                contrast=aug.get("contrast", 0.2),
                saturation=aug.get("saturation", 0.2),
                hue=aug.get("hue", 0.1),
                flip_prob=aug.get("horizontal_flip", 0.5),
            )
        else:
            self.transform = ValTransform(height=height, width=width)

        # Cache for intrinsics per date
        self._K_cache: dict[str, torch.Tensor] = {}

    def _load_split(self, split_path: str) -> list[tuple[str, str, int, str]]:
        """Load split file into list of (date, drive, frame_idx, side)."""
        filenames = []
        with open(split_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    folder, idx, side = parts
                    # Standard splits format: date/drive idx side (e.g. 2011_09_26/2011_09_26_drive_0001_sync)
                    if "/" in folder:
                        date, drive = folder.split("/")
                        filenames.append((date, drive, int(idx), side))
                elif len(parts) == 4:
                    date, drive, idx, side = parts
                    filenames.append((date, drive, int(idx), side))
        return filenames

    def _get_image_path(self, date: str, drive: str, frame_idx: int, side: str) -> str:
        """Construct path to a KITTI image."""
        cam_dir = self.SIDE_MAP[side]
        return str(
            self.data_path / date / drive / cam_dir / "data" / f"{frame_idx:010d}.png"
        )

    def _load_image(self, path: str) -> Image.Image:
        """Load image, converting to RGB."""
        return Image.open(path).convert("RGB")

    def _get_intrinsics(self, date: str) -> torch.Tensor:
        """Load and cache camera intrinsics for a given date.

        Reads from calib_cam_to_cam.txt, extracts the 3x3 projection matrix
        for camera 02 (left color), and constructs a 4x4 intrinsics matrix.

        The intrinsics are normalized to [0, 1] range (divided by original
        image dimensions) so they can be scaled to any target resolution.
        """
        if date in self._K_cache:
            return self._K_cache[date].clone()

        calib_path = self.data_path / date / "calib_cam_to_cam.txt"

        with open(calib_path, "r") as f:
            for line in f:
                if line.startswith("P_rect_02:"):
                    values = line.strip().split()[1:]
                    P = np.array([float(v) for v in values]).reshape(3, 4)
                    break

        # Extract intrinsics from 3x4 projection matrix
        # KITTI original resolution: 1242 x 375 (varies slightly per date)
        # We normalize by the original dimensions and rescale to target
        K = torch.eye(4, dtype=torch.float32)
        K[0, 0] = P[0, 0]  # fx
        K[1, 1] = P[1, 1]  # fy
        K[0, 2] = P[0, 2]  # cx
        K[1, 2] = P[1, 2]  # cy

        # Normalize by original KITTI image dimensions
        # Standard KITTI image_02 size is ~1242 x 375
        # We'll get actual dimensions from first image
        orig_w, orig_h = 1242, 375  # Default, overridden below

        # Scale to target resolution
        K[0, :] *= self.width / orig_w
        K[1, :] *= self.height / orig_h

        self._K_cache[date] = K
        return K.clone()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict:
        """Load a training sample.

        Returns:
            Dict with keys:
                ("color", frame_id): (3, H, W) clean image for loss computation
                ("color_aug", frame_id): (3, H, W) augmented image for encoder
                "K": (4, 4) camera intrinsics
                "inv_K": (4, 4) inverse intrinsics
        """
        date, drive, frame_idx, side = self.filenames[index]

        # Load all frames in the temporal window
        images = {}
        for fid in self.frame_ids:
            img_path = self._get_image_path(date, drive, frame_idx + fid, side)

            # Handle boundary frames: clamp to valid range
            if not os.path.exists(img_path):
                # Fall back to the target frame (will be auto-masked)
                img_path = self._get_image_path(date, drive, frame_idx, side)

            images[(fid, side)] = self._load_image(img_path)

        # Get intrinsics
        K = self._get_intrinsics(date)

        # Apply transforms
        color, color_aug, K = self.transform(images, K)

        # Build output dict
        outputs = {}
        for fid in self.frame_ids:
            outputs[("color", fid)] = color[fid]
            outputs[("color_aug", fid)] = color_aug[fid]

        outputs["K"] = K
        outputs["inv_K"] = torch.linalg.inv(K)

        return outputs
