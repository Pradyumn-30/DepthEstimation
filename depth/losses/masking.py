"""Auto-masking and minimum reprojection for handling occlusions and static pixels.

Two key innovations from Monodepth2 that dramatically improve training stability:

1. **Minimum reprojection**: For each pixel, take the minimum photometric loss
   across source frames. This handles occlusion — a pixel occluded in one
   source frame will have a valid match in another.

2. **Auto-masking**: Mask out pixels where the warped image is NOT better than
   the original (unwarped) source. This handles:
   - Static scenes where the camera doesn't move (loss would be zero regardless)
   - Objects moving at the same speed as the camera (appear stationary)
"""

import torch


def compute_min_reprojection(
    reprojection_losses: list[torch.Tensor],
) -> torch.Tensor:
    """Take per-pixel minimum across source frame reprojection losses.

    Args:
        reprojection_losses: List of (B, 1, H, W) loss tensors,
            one per source frame.

    Returns:
        (B, 1, H, W) per-pixel minimum loss.
    """
    stacked = torch.cat(reprojection_losses, dim=1)  # (B, N, H, W)
    min_loss, _ = stacked.min(dim=1, keepdim=True)  # (B, 1, H, W)
    return min_loss


def apply_auto_mask(
    reprojection_loss: torch.Tensor,
    identity_reprojection_losses: list[torch.Tensor],
) -> torch.Tensor:
    """Apply auto-masking: ignore pixels that don't benefit from warping.

    For each pixel, if the photometric error of the unwarped source image
    is lower than the warped image, it means warping made things worse —
    likely a static pixel or an object matching camera motion. Mask it out.

    We add a small random noise to the identity loss to break ties
    (ensures the loss is strictly less-than, not equal-to, which handles
    perfectly static frames).

    Args:
        reprojection_loss: (B, 1, H, W) warped photometric loss.
        identity_reprojection_losses: List of (B, 1, H, W) identity
            (unwarped) photometric losses, one per source frame.

    Returns:
        (B, 1, H, W) masked reprojection loss (zero where auto-masked).
    """
    # Identity reprojection: how well does the unwarped source match the target?
    identity_min = compute_min_reprojection(identity_reprojection_losses)

    # Add small random noise to break ties at exactly zero motion
    identity_min = identity_min + torch.randn_like(identity_min) * 1e-5

    # Mask: only keep pixels where warping is better than identity
    mask = (reprojection_loss < identity_min).float()

    return reprojection_loss * mask
