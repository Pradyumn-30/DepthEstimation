"""Edge-aware smoothness loss for disparity maps.

Encourages smooth depth in texture-less regions while allowing sharp
discontinuities at image edges (where depth boundaries are likely).
"""

import torch


def compute_smoothness_loss(
    disp: torch.Tensor, image: torch.Tensor
) -> torch.Tensor:
    """Edge-aware disparity smoothness loss.

    L_smooth = |∂d/∂x| * e^{-|∂I/∂x|} + |∂d/∂y| * e^{-|∂I/∂y|}

    The disparity gradients are penalized, but the penalty decays
    exponentially where the image has strong gradients (edges).

    The disparity is mean-normalized to maintain scale invariance:
    d_normalized = d / mean(d).

    Args:
        disp: (B, 1, H, W) disparity map.
        image: (B, 3, H, W) corresponding RGB image.

    Returns:
        Scalar smoothness loss (mean over all pixels and batch).
    """
    # Mean-normalize disparity for scale invariance
    mean_disp = disp.mean(dim=[2, 3], keepdim=True)
    norm_disp = disp / (mean_disp + 1e-7)

    # Disparity gradients
    disp_dx = torch.abs(norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:])
    disp_dy = torch.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])

    # Image gradients (mean across channels)
    image_dx = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean(
        dim=1, keepdim=True
    )
    image_dy = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(
        dim=1, keepdim=True
    )

    # Edge-aware weighting
    disp_dx = disp_dx * torch.exp(-image_dx)
    disp_dy = disp_dy * torch.exp(-image_dy)

    return disp_dx.mean() + disp_dy.mean()
