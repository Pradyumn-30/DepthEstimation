"""3D projection utilities: depth ↔ 3D point cloud ↔ 2D pixel coordinates.

These are the geometric primitives that enable view synthesis:
1. BackprojectDepth: pixel + depth → 3D point (using inverse intrinsics)
2. Project3D: 3D point → pixel in a new viewpoint (using intrinsics + pose)

Both operate in batch mode with no Python loops over pixels.
"""

import torch
import torch.nn as nn


class BackprojectDepth(nn.Module):
    """Unproject a depth map into a 3D point cloud.

    Creates a meshgrid of pixel coordinates once, then reuses it for every
    forward pass. Converts (u, v, depth) → (X, Y, Z, 1) in camera frame.

    Args:
        batch_size: Expected batch size (for pre-allocating meshgrid).
        height: Image height.
        width: Image width.
    """

    def __init__(self, batch_size: int, height: int, width: int):
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        # Create meshgrid of pixel coordinates: (1, 3, H*W)
        # Each column is [u, v, 1]^T (homogeneous pixel coords)
        meshgrid = torch.meshgrid(
            torch.arange(width, dtype=torch.float32),
            torch.arange(height, dtype=torch.float32),
            indexing="xy",
        )
        # meshgrid[0] = u coords (W along columns), meshgrid[1] = v coords (H along rows)
        id_coords = torch.stack([meshgrid[0], meshgrid[1]], dim=0)  # (2, H, W)

        ones = torch.ones(1, 1, height * width)
        pix_coords = torch.stack(
            [id_coords[0].reshape(-1), id_coords[1].reshape(-1)], dim=0
        )  # (2, H*W)
        pix_coords = pix_coords.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, H*W)
        pix_coords = torch.cat(
            [pix_coords, ones.repeat(batch_size, 1, 1)], dim=1
        )  # (B, 3, H*W)

        self.register_buffer("pix_coords", pix_coords)

    def forward(
        self, depth: torch.Tensor, inv_K: torch.Tensor
    ) -> torch.Tensor:
        """Backproject depth map to 3D points.

        Args:
            depth: (B, 1, H, W) depth map.
            inv_K: (B, 4, 4) inverse camera intrinsics.

        Returns:
            (B, 4, H*W) homogeneous 3D points in camera frame.
        """
        # inv_K[:, :3, :3] @ pix_coords → normalized camera rays
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)  # (B, 3, H*W)

        # Scale rays by depth
        cam_points = depth.reshape(depth.shape[0], 1, -1) * cam_points  # (B, 3, H*W)

        # Append homogeneous coordinate
        ones = torch.ones(
            depth.shape[0], 1, cam_points.shape[2],
            device=depth.device, dtype=depth.dtype
        )
        cam_points = torch.cat([cam_points, ones], dim=1)  # (B, 4, H*W)

        return cam_points


class Project3D(nn.Module):
    """Project 3D points to 2D pixel coordinates in a new viewpoint.

    Applies a 4x4 transformation (pose), then projects via camera intrinsics.
    Output coordinates are normalized to [-1, 1] for use with grid_sample.

    Args:
        batch_size: Expected batch size.
        height: Image height.
        width: Image width.
    """

    def __init__(self, batch_size: int, height: int, width: int):
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

    def forward(
        self,
        points_3d: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Project 3D points to normalized 2D coordinates.

        Args:
            points_3d: (B, 4, H*W) homogeneous 3D points.
            K: (B, 4, 4) camera intrinsics.
            T: (B, 4, 4) camera-to-camera transformation matrix.

        Returns:
            (B, H, W, 2) normalized pixel coordinates in [-1, 1],
            suitable for F.grid_sample.
        """
        # Transform points: K @ T @ points_3d
        P = torch.matmul(K, T)[:, :3, :]  # (B, 3, 4)
        cam_points = torch.matmul(P, points_3d)  # (B, 3, H*W)

        # Perspective divide
        pix_coords = cam_points[:, :2, :] / (
            cam_points[:, 2:3, :] + 1e-7
        )  # (B, 2, H*W)

        # Normalize to [-1, 1] for grid_sample
        pix_coords = pix_coords.reshape(
            pix_coords.shape[0], 2, self.height, self.width
        )
        pix_coords = pix_coords.permute(0, 2, 3, 1)  # (B, H, W, 2)

        # Normalize: u / (W-1) * 2 - 1, v / (H-1) * 2 - 1
        pix_coords[..., 0] = pix_coords[..., 0] / (self.width - 1) * 2 - 1
        pix_coords[..., 1] = pix_coords[..., 1] / (self.height - 1) * 2 - 1

        return pix_coords
