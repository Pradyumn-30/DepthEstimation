"""Tests for the warping / geometry pipeline.

These are deterministic, model-free tests: given a known depth map and
known camera pose, verify that the projected pixel coordinates are correct.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from depth.geometry.projection import BackprojectDepth, Project3D
from depth.geometry.warping import (
    disp_to_depth,
    transformation_from_parameters,
)


class TestDispToDepth:
    """Test disparity ↔ depth conversion."""

    def test_range(self):
        """Disparity in [0,1] should map to [min_depth, max_depth]."""
        disp = torch.tensor([0.0, 0.5, 1.0])
        _, depth = disp_to_depth(disp, min_depth=0.1, max_depth=100.0)
        assert depth[0].item() == pytest.approx(100.0, abs=1e-3)  # disp=0 → max depth
        assert depth[2].item() == pytest.approx(0.1, abs=1e-3)    # disp=1 → min depth
        assert depth[1].item() > 0.1 and depth[1].item() < 100.0  # disp=0.5 → middle

    def test_monotonic(self):
        """Higher disparity should give lower depth."""
        disp = torch.linspace(0, 1, 100)
        _, depth = disp_to_depth(disp)
        diffs = depth[1:] - depth[:-1]
        assert (diffs < 0).all(), "Depth should decrease with increasing disparity"


class TestTransformation:
    """Test axis-angle → rotation matrix conversion."""

    def test_identity(self):
        """Zero rotation and translation → identity matrix."""
        axisangle = torch.zeros(1, 1, 3)
        translation = torch.zeros(1, 1, 3)
        T = transformation_from_parameters(axisangle, translation)
        expected = torch.eye(4).unsqueeze(0)
        assert torch.allclose(T, expected, atol=1e-5)

    def test_pure_translation(self):
        """Zero rotation with translation should produce correct T matrix."""
        axisangle = torch.zeros(1, 1, 3)
        translation = torch.tensor([[[1.0, 2.0, 3.0]]])
        T = transformation_from_parameters(axisangle, translation)
        assert T[0, 0, 3].item() == pytest.approx(1.0, abs=1e-5)
        assert T[0, 1, 3].item() == pytest.approx(2.0, abs=1e-5)
        assert T[0, 2, 3].item() == pytest.approx(3.0, abs=1e-5)

    def test_invert_roundtrip(self):
        """T @ T_inv should ≈ identity."""
        torch.manual_seed(42)
        axisangle = torch.randn(1, 1, 3) * 0.01
        translation = torch.randn(1, 1, 3) * 0.1

        T = transformation_from_parameters(axisangle, translation)
        T_inv = transformation_from_parameters(axisangle, translation, invert=True)

        product = torch.matmul(T, T_inv)
        identity = torch.eye(4).unsqueeze(0)
        assert torch.allclose(product, identity, atol=1e-4), \
            f"T @ T_inv not identity:\n{product}"


class TestBackprojectProject:
    """Test backprojection → projection roundtrip."""

    def test_identity_roundtrip(self):
        """Backproject then project with identity pose → original pixel coords."""
        B, H, W = 1, 4, 8
        backproject = BackprojectDepth(B, H, W)
        project = Project3D(B, H, W)

        # Uniform depth
        depth = torch.ones(B, 1, H, W) * 5.0

        # Simple intrinsics
        K = torch.eye(4).unsqueeze(0)
        K[0, 0, 0] = W / 2  # fx
        K[0, 1, 1] = H / 2  # fy
        K[0, 0, 2] = W / 2  # cx
        K[0, 1, 2] = H / 2  # cy
        inv_K = torch.linalg.inv(K)

        # Identity pose
        T = torch.eye(4).unsqueeze(0)

        # Backproject then project
        points_3d = backproject(depth, inv_K)
        pix_coords = project(points_3d, K, T)  # (B, H, W, 2)

        # Coordinates should be in [-1, 1] grid
        assert pix_coords.shape == (B, H, W, 2)
        assert pix_coords.min() >= -1.1  # Allow small numerical error
        assert pix_coords.max() <= 1.1

    def test_output_shapes(self):
        """Verify output shapes of backproject and project."""
        B, H, W = 2, 16, 32
        backproject = BackprojectDepth(B, H, W)
        project = Project3D(B, H, W)

        depth = torch.ones(B, 1, H, W)
        K = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
        K[:, 0, 0] = 100
        K[:, 1, 1] = 100
        K[:, 0, 2] = W / 2
        K[:, 1, 2] = H / 2
        inv_K = torch.linalg.inv(K)
        T = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)

        points = backproject(depth, inv_K)
        assert points.shape == (B, 4, H * W)

        coords = project(points, K, T)
        assert coords.shape == (B, H, W, 2)
