"""Tests for the KITTI dataloader.

These tests verify the dataloader produces correct shapes, data types,
and intrinsics properties. They require actual KITTI data to be downloaded.
Tests are skipped if data is not available.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

KITTI_DATA = Path("data/kitti_raw")
SPLIT_FILE = Path("data/splits/eigen_curated/train_files.txt")


@pytest.fixture
def dataset():
    """Create a small KITTI dataset."""
    from depth.datasets.kitti_dataset import KITTIDataset

    return KITTIDataset(
        data_path=str(KITTI_DATA),
        split_path=str(SPLIT_FILE),
        height=128,
        width=416,
        frame_ids=[0, -1, 1],
        is_train=True,
    )


@pytest.mark.skipif(
    not KITTI_DATA.exists() or not SPLIT_FILE.exists(),
    reason="KITTI data not downloaded"
)
class TestKITTIDataset:

    def test_length(self, dataset):
        """Dataset should have non-zero length."""
        assert len(dataset) > 0

    def test_sample_keys(self, dataset):
        """Each sample should contain expected keys."""
        sample = dataset[0]
        expected_keys = [
            ("color", 0), ("color", -1), ("color", 1),
            ("color_aug", 0), ("color_aug", -1), ("color_aug", 1),
            "K", "inv_K",
        ]
        for key in expected_keys:
            assert key in sample, f"Missing key: {key}"

    def test_image_shapes(self, dataset):
        """Images should be (3, H, W) tensors in [0, 1]."""
        sample = dataset[0]
        for fid in [0, -1, 1]:
            img = sample[("color", fid)]
            assert img.shape == (3, 128, 416), f"Wrong shape: {img.shape}"
            assert img.dtype == torch.float32
            assert img.min() >= 0.0
            assert img.max() <= 1.0

    def test_intrinsics(self, dataset):
        """Intrinsics should be valid (positive focal lengths, etc.)."""
        sample = dataset[0]
        K = sample["K"]
        assert K.shape == (4, 4)
        assert K[0, 0] > 0, "fx should be positive"
        assert K[1, 1] > 0, "fy should be positive"
        assert K[0, 2] > 0, "cx should be positive"
        assert K[1, 2] > 0, "cy should be positive"
        assert K[3, 3] == 1.0, "Homogeneous coord should be 1"

    def test_inverse_intrinsics(self, dataset):
        """K @ inv_K should ≈ identity."""
        sample = dataset[0]
        product = sample["K"] @ sample["inv_K"]
        identity = torch.eye(4)
        assert torch.allclose(product, identity, atol=1e-4), \
            f"K @ inv_K != I:\n{product}"

    def test_official_eigen_split(self):
        """Verify the dataloader can parse the 3-part official Eigen split format."""
        from depth.datasets.kitti_dataset import KITTIDataset
        official_split_file = Path("data/splits/eigen/test_files.txt")
        if not official_split_file.exists():
            pytest.skip("Official Eigen split file not downloaded")
            
        dataset = KITTIDataset(
            data_path=str(KITTI_DATA),
            split_path=str(official_split_file),
            height=128,
            width=416,
            frame_ids=[0],
            is_train=False,
        )
        assert len(dataset.filenames) == 697
        # Verify first entry was parsed into 4 parts: (date, drive, idx, side)
        first_entry = dataset.filenames[0]
        assert len(first_entry) == 4
        assert "/" not in first_entry[0]
        assert "/" not in first_entry[1]
        assert isinstance(first_entry[2], int)
        assert first_entry[3] in ["l", "r"]

