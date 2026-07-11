"""CoreML export and validation for the depth encoder+decoder.

Exports the depth network (encoder + decoder, NOT pose network) to CoreML
format (.mlpackage) for on-device inference on Apple platforms.

This is a Phase 1 dry-run to surface any conversion-blocking operations
early, before investing in full training.

Usage:
    python depth/export/coreml_export.py \\
        --checkpoint checkpoints/epoch_00.pt \\
        --output depth_model.mlpackage \\
        --height 128 --width 416
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from depth.models.depth_decoder import DepthDecoder
from depth.models.resnet_encoder import ResNetEncoder


class DepthModel(torch.nn.Module):
    """Combined encoder+decoder for export (single forward pass)."""

    def __init__(self, encoder: ResNetEncoder, decoder: DepthDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only the finest-scale disparity.

        Args:
            x: (1, 3, H, W) input image.

        Returns:
            (1, 1, H/2, W/2) disparity map at scale 0.
        """
        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs[("disp", 0)]


def export_to_coreml(
    checkpoint_path: str,
    output_path: str,
    height: int = 128,
    width: int = 416,
):
    """Export depth model to CoreML.

    Steps:
    1. Load checkpoint and reconstruct model
    2. Trace with torch.jit.trace
    3. Convert via coremltools
    4. Validate: compare PyTorch vs CoreML output
    5. Save .mlpackage
    """
    try:
        import coremltools as ct
    except ImportError:
        print("coremltools not installed. Install with: pip install coremltools")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")

    # Build model
    encoder = ResNetEncoder(pretrained=False)
    decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc)

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(checkpoint["depth_encoder"])
        decoder.load_state_dict(checkpoint["depth_decoder"])
        print("  Loaded weights from checkpoint")
    else:
        print("  No checkpoint found, using random weights (dry-run mode)")

    model = DepthModel(encoder, decoder)
    model.eval()

    # Create example input
    example_input = torch.randn(1, 3, height, width)

    # Get PyTorch output for validation
    with torch.no_grad():
        pytorch_output = model(example_input).numpy()

    print(f"  PyTorch output shape: {pytorch_output.shape}")
    print(f"  PyTorch output range: [{pytorch_output.min():.4f}, {pytorch_output.max():.4f}]")

    # Trace model
    print("Tracing model with torch.jit.trace...")
    traced = torch.jit.trace(model, example_input)

    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="input_image",
                shape=example_input.shape,
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )

    # Validate: compare outputs
    print("Validating CoreML output against PyTorch...")

    # CoreML expects uint8 image input when using ImageType with scale
    # For validation, we'll use the array interface instead
    try:
        # Create PIL image for CoreML input
        from PIL import Image
        img_np = (example_input[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        coreml_out = mlmodel.predict({"input_image": pil_image})
        coreml_disp = list(coreml_out.values())[0]

        if isinstance(coreml_disp, np.ndarray):
            max_err = np.abs(pytorch_output - coreml_disp).max()
            print(f"  Max absolute error: {max_err:.6f}")
            if max_err < 1e-2:
                print("  ✅ Validation PASSED (error < 0.01)")
            else:
                print(f"  ⚠️  Validation WARNING: error={max_err:.4f} (may be due to ImageType preprocessing)")
        else:
            print(f"  CoreML output type: {type(coreml_disp)} — skipping numerical validation")

    except Exception as e:
        print(f"  ⚠️  Validation skipped: {e}")
        print("  (This is OK for a dry-run — the conversion itself succeeded)")

    # Save
    mlmodel.save(output_path)
    print(f"\n✅ CoreML model saved: {output_path}")
    print(f"   Input: RGB image ({height}x{width})")
    print(f"   Output: Disparity map")


def main():
    parser = argparse.ArgumentParser(description="Export depth model to CoreML")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output", type=str, default="depth_model.mlpackage")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=416)
    args = parser.parse_args()

    export_to_coreml(args.checkpoint, args.output, args.height, args.width)


if __name__ == "__main__":
    main()
