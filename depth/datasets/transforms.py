"""Data augmentation transforms for KITTI training.

Applied on-the-fly during training. Two types:
1. Geometric (horizontal flip) — applied to both image AND intrinsics
2. Color (jitter) — applied only to the augmented copy, not the clean copy
   used for loss computation (avoids penalizing color-invariant predictions)
"""

import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class TrainTransform:
    """Training augmentations: color jitter + random horizontal flip.

    Color jitter is applied to produce an augmented image used as encoder input,
    while the clean image is used for photometric loss computation.
    This prevents the network from learning to predict color-dependent depth.

    Args:
        height: Target height after resize.
        width: Target width after resize.
        brightness: Max brightness jitter.
        contrast: Max contrast jitter.
        saturation: Max saturation jitter.
        hue: Max hue jitter.
        flip_prob: Probability of horizontal flip.
    """

    def __init__(
        self,
        height: int,
        width: int,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        flip_prob: float = 0.5,
    ):
        self.height = height
        self.width = width
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.flip_prob = flip_prob

    def __call__(
        self, images: dict[tuple, Image.Image], K: torch.Tensor
    ) -> tuple[dict, dict, torch.Tensor]:
        """Apply transforms to a set of images and intrinsics.

        Args:
            images: Dict mapping (frame_id, side) → PIL Image.
                e.g., {(-1, "l"): img, (0, "l"): img, (1, "l"): img}
            K: (4, 4) camera intrinsics matrix.

        Returns:
            Tuple of:
                - color: Dict mapping frame_id → (3, H, W) clean tensor
                - color_aug: Dict mapping frame_id → (3, H, W) augmented tensor
                - K: (4, 4) possibly flipped intrinsics
        """
        # Decide flip for this sample (same flip for all frames)
        do_flip = random.random() < self.flip_prob

        # Sample color jitter parameters (same for all frames in this sample)
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        saturation_factor = 1.0 + random.uniform(-self.saturation, self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)

        color = {}
        color_aug = {}

        for key, img in images.items():
            frame_id = key[0]

            # Resize
            img = img.resize((self.width, self.height), Image.LANCZOS)

            # Horizontal flip
            if do_flip:
                img = TF.hflip(img)

            # Clean version (for loss computation)
            color[frame_id] = TF.to_tensor(img)  # (3, H, W), [0, 1]

            # Augmented version (for encoder input)
            aug = TF.adjust_brightness(img, brightness_factor)
            aug = TF.adjust_contrast(aug, contrast_factor)
            aug = TF.adjust_saturation(aug, saturation_factor)
            aug = TF.adjust_hue(aug, hue_factor)
            color_aug[frame_id] = TF.to_tensor(aug)

        # Adjust intrinsics for flip
        if do_flip:
            K = K.clone()
            K[0, 2] = self.width - K[0, 2]  # Flip cx

        return color, color_aug, K


class ValTransform:
    """Validation/test transforms: resize only, no augmentation."""

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(
        self, images: dict[tuple, Image.Image], K: torch.Tensor
    ) -> tuple[dict, dict, torch.Tensor]:
        color = {}
        for key, img in images.items():
            frame_id = key[0]
            img = img.resize((self.width, self.height), Image.LANCZOS)
            color[frame_id] = TF.to_tensor(img)

        # For val, augmented = clean
        return color, color, K
