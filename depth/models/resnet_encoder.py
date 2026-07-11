"""ResNet18 encoder for depth and pose networks.

Extracts multi-scale features at 5 resolution levels from a pretrained
ResNet18 backbone. Used by both the depth decoder (via skip connections)
and the pose network.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """ResNet18 encoder that outputs features at 5 scales.

    Scale 0: stride 1  (H, W)     — after initial conv (before maxpool)
    Scale 1: stride 2  (H/2, W/2) — after maxpool
    Scale 2: stride 4  (H/4, W/4) — after layer1
    Scale 3: stride 8  (H/8, W/8) — after layer2
    Scale 4: stride 16 (H/16, W/16) — after layer3... wait, let me reconsider.

    Actually, ResNet18 has:
    - conv1 + bn1 + relu: stride 2 → (H/2, W/2), 64 channels
    - maxpool: stride 2 → (H/4, W/4)
    - layer1: stride 1 → (H/4, W/4), 64 channels
    - layer2: stride 2 → (H/8, W/8), 128 channels
    - layer3: stride 2 → (H/16, W/16), 256 channels
    - layer4: stride 2 → (H/32, W/32), 512 channels

    Monodepth2 convention for feature extraction:
    Scale 0: after relu (conv1+bn1+relu) → 64ch, (H/2, W/2)
    Scale 1: after layer1 → 64ch, (H/4, W/4)
    Scale 2: after layer2 → 128ch, (H/8, W/8)
    Scale 3: after layer3 → 256ch, (H/16, W/16)
    Scale 4: after layer4 → 512ch, (H/32, W/32)

    Args:
        pretrained: If True, loads ImageNet pretrained weights.
        num_input_images: Number of input images concatenated channel-wise.
            1 for depth encoder, 2 for pose encoder (stacked pair).
    """

    def __init__(self, pretrained: bool = True, num_input_images: int = 1):
        super().__init__()

        self.num_ch_enc = [64, 64, 128, 256, 512]

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # If pose encoder: modify first conv to accept 6-channel input (2 RGB images)
        if num_input_images != 1:
            self.conv1 = nn.Conv2d(
                num_input_images * 3, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize by averaging pretrained conv1 weights across input images
            if pretrained:
                old_weights = resnet.conv1.weight.data
                # Repeat and average: (64, 3, 7, 7) → (64, 6, 7, 7) for 2 images
                self.conv1.weight.data = old_weights.repeat(1, num_input_images, 1, 1)
                self.conv1.weight.data /= num_input_images
        else:
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor (B, C, H, W) where C=3 for depth, C=6 for pose.

        Returns:
            List of 5 feature tensors at increasing depth / decreasing resolution.
        """
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)          # Scale 0: 64ch, (H/2, W/2)

        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)          # Scale 1: 64ch, (H/4, W/4)

        x = self.layer2(x)
        features.append(x)          # Scale 2: 128ch, (H/8, W/8)

        x = self.layer3(x)
        features.append(x)          # Scale 3: 256ch, (H/16, W/16)

        x = self.layer4(x)
        features.append(x)          # Scale 4: 512ch, (H/32, W/32)

        return features
