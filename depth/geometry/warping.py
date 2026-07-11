"""Image warping via grid_sample for view synthesis.

This is the core operation that connects depth + pose to photometric loss:
given a source image, predicted depth of the target, and the relative pose,
synthesize what the target frame *should* look like from the source viewpoint.

The critical op here is F.grid_sample, which must run natively on MPS
(verified by scripts/verify_mps.py before training).
"""

import torch
import torch.nn.functional as F

from depth.geometry.projection import BackprojectDepth, Project3D


def disp_to_depth(disp: torch.Tensor, min_depth: float = 0.1, max_depth: float = 100.0):
    """Convert disparity to depth.

    depth = 1 / (a * disp + b) where a, b are chosen so that
    disp=0 → max_depth and disp=1 → min_depth.

    Args:
        disp: Disparity tensor in [0, 1] (sigmoid output).
        min_depth: Minimum depth value.
        max_depth: Maximum depth value.

    Returns:
        Tuple of (scaled_disp, depth).
    """
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(
    axisangle: torch.Tensor, translation: torch.Tensor, invert: bool = False
) -> torch.Tensor:
    """Convert axis-angle + translation to a 4x4 transformation matrix.

    Args:
        axisangle: (B, 1, 3) axis-angle rotation.
        translation: (B, 1, 3) translation vector.
        invert: If True, return the inverse transformation.

    Returns:
        (B, 4, 4) transformation matrix.
    """
    R = _rot_from_axisangle(axisangle)  # (B, 3, 3) via Rodrigues
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t = -1.0 * t

    T = _get_translation_matrix(t)  # (B, 4, 4)

    if invert:
        M = torch.matmul(T, _pad_rotation(R))
    else:
        M = torch.matmul(_pad_rotation(R), T)

    return M


def _rot_from_axisangle(vec: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to rotation matrix via Rodrigues' formula.

    Args:
        vec: (B, 1, 3) axis-angle vector.

    Returns:
        (B, 3, 3) rotation matrix.
    """
    angle = vec.norm(dim=2, keepdim=True).unsqueeze(2)  # (B, 1, 1, 1)
    axis = vec / (angle.squeeze(2) + 1e-7)  # (B, 1, 3)

    cos_a = torch.cos(angle)  # (B, 1, 1, 1)
    sin_a = torch.sin(angle)

    # Reshape for matrix construction
    cos_a = cos_a.squeeze()  # scalar-like per batch
    sin_a = sin_a.squeeze()

    # Use batch-compatible Rodrigues
    # For small batch: construct K (skew-symmetric) then R = I + sin(a)*K + (1-cos(a))*K^2
    ax = axis.squeeze(1)  # (B, 3)
    B = ax.shape[0]

    # Build rotation matrix directly from axis-angle components
    x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]
    ca, sa = cos_a.reshape(B), sin_a.reshape(B)
    t = 1.0 - ca

    # Rodrigues' rotation formula (explicit matrix elements)
    R = torch.zeros(B, 3, 3, device=vec.device, dtype=vec.dtype)
    R[:, 0, 0] = t * x * x + ca
    R[:, 0, 1] = t * x * y - sa * z
    R[:, 0, 2] = t * x * z + sa * y
    R[:, 1, 0] = t * x * y + sa * z
    R[:, 1, 1] = t * y * y + ca
    R[:, 1, 2] = t * y * z - sa * x
    R[:, 2, 0] = t * x * z - sa * y
    R[:, 2, 1] = t * y * z + sa * x
    R[:, 2, 2] = t * z * z + ca

    return R


def _get_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    """Create 4x4 translation matrix from (B, 1, 3) translation vector."""
    B = translation.shape[0]
    T = torch.eye(4, device=translation.device, dtype=translation.dtype)
    T = T.unsqueeze(0).repeat(B, 1, 1)

    t = translation.squeeze(1)  # (B, 3)
    T[:, 0, 3] = t[:, 0]
    T[:, 1, 3] = t[:, 1]
    T[:, 2, 3] = t[:, 2]

    return T


def _pad_rotation(R: torch.Tensor) -> torch.Tensor:
    """Pad 3x3 rotation to 4x4 homogeneous matrix."""
    B = R.shape[0]
    T = torch.zeros(B, 4, 4, device=R.device, dtype=R.dtype)
    T[:, :3, :3] = R
    T[:, 3, 3] = 1.0
    return T


def warp_image(
    source_image: torch.Tensor,
    depth: torch.Tensor,
    T: torch.Tensor,
    K: torch.Tensor,
    inv_K: torch.Tensor,
    backproject: BackprojectDepth,
    project: Project3D,
) -> torch.Tensor:
    """Synthesize target view by warping source image.

    Pipeline: depth → 3D points → transform by pose → project to 2D → sample.

    Args:
        source_image: (B, 3, H, W) source frame.
        depth: (B, 1, H, W) predicted depth of the target frame.
        T: (B, 4, 4) source-to-target camera transformation.
        K: (B, 4, 4) camera intrinsics.
        inv_K: (B, 4, 4) inverse camera intrinsics.
        backproject: BackprojectDepth module.
        project: Project3D module.

    Returns:
        (B, 3, H, W) synthesized target view.
    """
    # Step 1: Depth → 3D point cloud in target camera frame
    cam_points = backproject(depth, inv_K)  # (B, 4, H*W)

    # Step 2: Transform to source camera frame and project to 2D
    pix_coords = project(cam_points, K, T)  # (B, H, W, 2)

    # Step 3: Sample source image at projected coordinates
    warped = F.grid_sample(
        source_image,
        pix_coords,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return warped
