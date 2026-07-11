"""Tests for model forward pass.

Verifies output shapes and value ranges for all model components.
No trained weights needed — uses random initialization.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from depth.models.resnet_encoder import ResNetEncoder
from depth.models.depth_decoder import DepthDecoder
from depth.models.pose_decoder import PoseDecoder


class TestResNetEncoder:
    """Test the ResNet18 feature encoder."""

    def test_depth_encoder_shapes(self):
        """Depth encoder with 3-channel input should produce 5 feature maps."""
        encoder = ResNetEncoder(pretrained=False, num_input_images=1)
        x = torch.randn(2, 3, 128, 416)
        features = encoder(x)

        assert len(features) == 5
        expected_channels = [64, 64, 128, 256, 512]
        for i, (feat, ch) in enumerate(zip(features, expected_channels)):
            assert feat.shape[0] == 2, f"Batch dim wrong at scale {i}"
            assert feat.shape[1] == ch, f"Channel count wrong at scale {i}: {feat.shape[1]} != {ch}"

    def test_pose_encoder_shapes(self):
        """Pose encoder with 6-channel input (2 stacked images)."""
        encoder = ResNetEncoder(pretrained=False, num_input_images=2)
        x = torch.randn(2, 6, 128, 416)
        features = encoder(x)

        assert len(features) == 5
        assert features[0].shape[1] == 64
        assert features[-1].shape[1] == 512

    def test_spatial_downsampling(self):
        """Feature maps should progressively downsample."""
        encoder = ResNetEncoder(pretrained=False)
        x = torch.randn(1, 3, 128, 416)
        features = encoder(x)

        # Each feature should be spatially smaller than the previous
        for i in range(1, len(features)):
            assert features[i].shape[2] <= features[i - 1].shape[2]
            assert features[i].shape[3] <= features[i - 1].shape[3]


class TestDepthDecoder:
    """Test the multi-scale disparity decoder."""

    def test_output_shapes(self):
        """Decoder should produce disparity maps at 4 scales."""
        encoder = ResNetEncoder(pretrained=False)
        decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=[0, 1, 2, 3])

        x = torch.randn(2, 3, 128, 416)
        features = encoder(x)
        outputs = decoder(features)

        assert len(outputs) == 4
        for scale in [0, 1, 2, 3]:
            key = ("disp", scale)
            assert key in outputs, f"Missing output at scale {scale}"
            assert outputs[key].shape[0] == 2  # batch
            assert outputs[key].shape[1] == 1  # single channel

    def test_disparity_range(self):
        """Disparity values should be in [0, 1] (sigmoid output)."""
        encoder = ResNetEncoder(pretrained=False)
        decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc)

        x = torch.randn(2, 3, 128, 416)
        features = encoder(x)
        outputs = decoder(features)

        for key, disp in outputs.items():
            assert disp.min() >= 0.0, f"Disparity < 0 at {key}"
            assert disp.max() <= 1.0, f"Disparity > 1 at {key}"


class TestPoseDecoder:
    """Test the pose prediction network."""

    def test_output_shape(self):
        """Pose decoder should output (B, num_frames, 6)."""
        encoder = ResNetEncoder(pretrained=False, num_input_images=2)
        decoder = PoseDecoder(
            num_ch_enc=encoder.num_ch_enc,
            num_frames_to_predict_for=2,
        )

        x = torch.randn(2, 6, 128, 416)
        features = encoder(x)
        pose = decoder(features)

        assert pose.shape == (2, 2, 6), f"Wrong pose shape: {pose.shape}"

    def test_initial_near_zero(self):
        """Initial (untrained) pose predictions should be near zero (0.01 scaling)."""
        encoder = ResNetEncoder(pretrained=False, num_input_images=2)
        decoder = PoseDecoder(
            num_ch_enc=encoder.num_ch_enc,
            num_frames_to_predict_for=2,
        )

        x = torch.randn(1, 6, 128, 416)
        features = encoder(x)
        pose = decoder(features)

        # With 0.01 scaling, initial predictions should be small
        assert pose.abs().max() < 1.0, \
            f"Initial pose predictions too large: max={pose.abs().max():.4f}"


class TestEndToEnd:
    """Test the full depth pipeline end-to-end."""

    def test_full_pipeline_no_crash(self):
        """Full forward + backward pass should complete without error."""
        depth_enc = ResNetEncoder(pretrained=False, num_input_images=1)
        depth_dec = DepthDecoder(num_ch_enc=depth_enc.num_ch_enc)
        pose_enc = ResNetEncoder(pretrained=False, num_input_images=2)
        pose_dec = PoseDecoder(num_ch_enc=pose_enc.num_ch_enc)

        # Depth forward
        target = torch.randn(2, 3, 128, 416)
        features = depth_enc(target)
        depth_out = depth_dec(features)

        # Pose forward
        source = torch.randn(2, 3, 128, 416)
        pose_input = torch.cat([source, target], dim=1)
        pose_features = pose_enc(pose_input)
        pose_out = pose_dec(pose_features)

        # Simple loss and backward
        loss = sum(v.mean() for v in depth_out.values()) + pose_out.mean()
        loss.backward()

        # Verify gradients exist
        for p in depth_enc.parameters():
            if p.requires_grad:
                assert p.grad is not None, "Missing gradient in depth encoder"
                break
