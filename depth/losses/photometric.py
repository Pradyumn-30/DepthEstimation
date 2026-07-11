"""Photometric reprojection loss: weighted combination of L1 and SSIM.

The core self-supervised signal: how well does the warped source image
match the actual target image? Lower = better depth + pose prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    """Structural Similarity Index (SSIM) loss.

    Uses a 3x3 mean filter (not the full 11x11 Gaussian from the original
    SSIM paper) following Monodepth2's implementation for efficiency.
    """

    def __init__(self):
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between x and y.

        Args:
            x, y: (B, 3, H, W) image tensors in [0, 1].

        Returns:
            (B, 1, H-2, W-2) per-pixel SSIM loss (1 - SSIM)/2, range [0, 1].
            The spatial dimensions shrink by 2 due to the 3x3 pooling without padding.
        """
        # Pad to maintain spatial dimensions
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        y = F.pad(y, (1, 1, 1, 1), mode="reflect")

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        ssim = ssim_n / ssim_d  # (B, C, H, W)

        return torch.clamp((1 - ssim.mean(dim=1, keepdim=True)) / 2, 0, 1)


def compute_reprojection_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ssim_module: SSIM,
    ssim_weight: float = 0.85,
) -> torch.Tensor:
    """Compute photometric reprojection loss (L1 + SSIM).

    Args:
        pred: (B, 3, H, W) warped/synthesized image.
        target: (B, 3, H, W) actual target image.
        ssim_module: Pre-initialized SSIM module.
        ssim_weight: Weight for SSIM term (1 - ssim_weight for L1).

    Returns:
        (B, 1, H, W) per-pixel reprojection loss.
    """
    l1_loss = torch.abs(pred - target).mean(dim=1, keepdim=True)
    ssim_loss = ssim_module(pred, target)

    loss = ssim_weight * ssim_loss + (1 - ssim_weight) * l1_loss

    return loss
