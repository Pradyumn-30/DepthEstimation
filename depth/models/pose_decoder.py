"""Pose decoder: predicts 6-DOF relative camera pose between frame pairs.

Takes encoder features from a pair of concatenated images and outputs
axis-angle rotation (3) + translation (3) for the relative transform.
"""

import torch
import torch.nn as nn


class PoseDecoder(nn.Module):
    """Predicts relative camera pose from encoder features.

    Following Monodepth2: takes the last encoder feature map, applies
    a series of 1x1 convolutions to compress, then predicts 6-DOF pose.

    The output is scaled by 0.01 so initial predictions are near-identity,
    which stabilizes early training.

    Args:
        num_ch_enc: List of encoder channel counts (only last is used).
        num_input_features: Number of feature sets concatenated (typically 1,
            since we use a separate pose encoder that takes stacked images).
        num_frames_to_predict_for: Number of relative poses to predict.
            For frame_ids [0, -1, 1], this is 2 (pose from -1→0 and 1→0).
    """

    def __init__(
        self,
        num_ch_enc: list[int],
        num_input_features: int = 1,
        num_frames_to_predict_for: int = 2,
    ):
        super().__init__()

        self.num_frames_to_predict_for = num_frames_to_predict_for

        # Squeeze: reduce channels from encoder
        self.squeeze = nn.Conv2d(
            num_ch_enc[-1] * num_input_features, 256, kernel_size=1
        )
        self.relu = nn.ReLU(inplace=True)

        # Pose prediction: 1x1 convolutions → 6-DOF output
        self.pose_conv0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pose_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pose_conv2 = nn.Conv2d(
            256, 6 * num_frames_to_predict_for, kernel_size=1
        )

    def forward(self, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        """Predict relative poses from encoder features.

        Args:
            encoder_features: List of feature tensors from pose encoder.
                Only the last (deepest) feature map is used.

        Returns:
            Pose tensor of shape (B, num_frames, 6) where the 6 values are
            [ax_angle_x, ax_angle_y, ax_angle_z, tx, ty, tz].
            Scaled by 0.01 for training stability.
        """
        # Use only the deepest feature map
        x = encoder_features[-1]

        x = self.relu(self.squeeze(x))
        x = self.relu(self.pose_conv0(x))
        x = self.relu(self.pose_conv1(x))
        x = self.pose_conv2(x)

        # Global average pool → (B, 6*num_frames, 1, 1) → (B, 6*num_frames)
        x = x.mean(dim=[2, 3])

        # Reshape to (B, num_frames, 6)
        x = x.view(-1, self.num_frames_to_predict_for, 6)

        # Scale to keep initial predictions near identity
        x = 0.01 * x

        return x
