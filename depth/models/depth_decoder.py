"""Depth decoder: U-Net style decoder producing multi-scale disparity maps.

Takes multi-scale encoder features and produces disparity (inverse depth)
at 4 spatial resolutions via skip connections and progressive upsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ConvBlock(nn.Module):
    """Conv3x3 → ELU block (no batch norm, following Monodepth2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu(self.conv(x))


class DepthDecoder(nn.Module):
    """Multi-scale disparity decoder with skip connections.

    Decodes encoder features into disparity maps at 4 scales.
    Disparity is converted to depth via: depth = 1 / (a * sigmoid(d) + b)

    Args:
        num_ch_enc: List of channel counts from encoder at each scale.
            Default for ResNet18: [64, 64, 128, 256, 512]
        num_output_channels: Number of disparity channels (1 for monocular).
        scales: Which scales to output disparity at. Default [0,1,2,3].
    """

    def __init__(
        self,
        num_ch_enc: list[int] = None,
        num_output_channels: int = 1,
        scales: list[int] = None,
    ):
        super().__init__()

        if num_ch_enc is None:
            num_ch_enc = [64, 64, 128, 256, 512]
        if scales is None:
            scales = [0, 1, 2, 3]

        self.num_output_channels = num_output_channels
        self.scales = scales

        # Decoder channel widths at each level (from deepest to shallowest)
        num_ch_dec = [16, 32, 64, 128, 256]

        self.convs = OrderedDict()

        for i in range(4, -1, -1):  # 4, 3, 2, 1, 0
            # Upconv 0: reduce channels from encoder/previous decoder level
            if i == 4:
                num_ch_in = num_ch_enc[i]
            else:
                num_ch_in = num_ch_dec[i + 1]
            num_ch_out = num_ch_dec[i]
            self.convs[f"upconv_{i}_0"] = ConvBlock(num_ch_in, num_ch_out)

            # Upconv 1: after concatenation with skip connection
            if i > 0:
                num_ch_in = num_ch_dec[i] + num_ch_enc[i - 1]
            else:
                num_ch_in = num_ch_dec[i]
            num_ch_out = num_ch_dec[i]
            self.convs[f"upconv_{i}_1"] = ConvBlock(num_ch_in, num_ch_out)

        # Disparity output convolutions at each requested scale
        for s in self.scales:
            self.convs[f"dispconv_{s}"] = nn.Conv2d(
                num_ch_dec[s], self.num_output_channels, 3, padding=1
            )

        self.decoder = nn.ModuleList(list(self.convs.values()))
        # Keep ordered dict for name-based access
        self.convs = nn.ModuleDict(self.convs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_features: list[torch.Tensor]) -> dict[tuple, torch.Tensor]:
        """Decode encoder features into multi-scale disparity maps.

        Args:
            encoder_features: List of 5 tensors from ResNetEncoder.

        Returns:
            Dict mapping ("disp", scale) → disparity tensor.
            Disparity values are in [0, 1] via sigmoid.
        """
        outputs = {}
        x = encoder_features[-1]  # Start from deepest features

        for i in range(4, -1, -1):
            x = self.convs[f"upconv_{i}_0"](x)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

            # Skip connection (concat with encoder features from one level shallower)
            if i > 0:
                x = torch.cat([x, encoder_features[i - 1]], dim=1)

            x = self.convs[f"upconv_{i}_1"](x)

            # Output disparity at this scale if requested
            if i in self.scales:
                disp = self.sigmoid(self.convs[f"dispconv_{i}"](x))
                outputs[("disp", i)] = disp

        return outputs
