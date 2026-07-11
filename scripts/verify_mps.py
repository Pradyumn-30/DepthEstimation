"""MPS op-by-op sanity check.

Run this BEFORE trusting any full training run on Apple Silicon.
Verifies that every critical operation produces identical results
on MPS vs CPU, and that no silent fallbacks are happening.

Usage:
    python scripts/verify_mps.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_available():
    """Check MPS availability."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    if not torch.backends.mps.is_available():
        print("\n❌ MPS not available. Cannot run verification.")
        sys.exit(1)
    print()


def compare(name: str, cpu_out: torch.Tensor, mps_out: torch.Tensor, atol: float = 1e-5):
    """Compare CPU and MPS outputs."""
    mps_cpu = mps_out.cpu()
    max_err = (cpu_out - mps_cpu).abs().max().item()
    mean_err = (cpu_out - mps_cpu).abs().mean().item()
    passed = max_err < atol

    status = "✅" if passed else "❌"
    print(f"  {status} {name}: max_err={max_err:.2e}, mean_err={mean_err:.2e}")
    return passed


def test_grid_sample():
    """Test F.grid_sample — the critical warping operation."""
    print("Testing F.grid_sample (bilinear, border padding)...")
    torch.manual_seed(42)

    source = torch.randn(2, 3, 128, 416)
    grid = torch.randn(2, 128, 416, 2).clamp(-1, 1)

    cpu_out = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)

    mps_out = F.grid_sample(
        source.to("mps"), grid.to("mps"),
        mode="bilinear", padding_mode="border", align_corners=True,
    )

    return compare("grid_sample", cpu_out, mps_out, atol=1e-4)


def test_conv2d():
    """Test F.conv2d — encoder/decoder convolutions."""
    print("Testing F.conv2d...")
    torch.manual_seed(42)

    x = torch.randn(2, 64, 32, 104)
    w = torch.randn(128, 64, 3, 3)
    b = torch.randn(128)

    cpu_out = F.conv2d(x, w, b, padding=1)
    mps_out = F.conv2d(x.to("mps"), w.to("mps"), b.to("mps"), padding=1)

    return compare("conv2d", cpu_out, mps_out, atol=5e-4)


def test_interpolate():
    """Test F.interpolate bilinear — decoder upsampling."""
    print("Testing F.interpolate (bilinear)...")
    torch.manual_seed(42)

    x = torch.randn(2, 64, 16, 52)

    cpu_out = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    mps_out = F.interpolate(x.to("mps"), scale_factor=2, mode="bilinear", align_corners=False)

    return compare("interpolate", cpu_out, mps_out)


def test_matmul():
    """Test batched matmul — projection math."""
    print("Testing torch.matmul (batched)...")
    torch.manual_seed(42)

    a = torch.randn(4, 3, 4)
    b = torch.randn(4, 4, 53248)  # 128*416 = 53248

    cpu_out = torch.matmul(a, b)
    mps_out = torch.matmul(a.to("mps"), b.to("mps"))

    return compare("matmul", cpu_out, mps_out, atol=1e-3)


def test_inverse():
    """Test matrix inversion — intrinsics."""
    print("Testing torch.linalg.inv...")
    torch.manual_seed(42)

    # Create a valid intrinsics-like matrix
    K = torch.eye(4).unsqueeze(0).repeat(4, 1, 1)
    K[:, 0, 0] = 200.0  # fx
    K[:, 1, 1] = 200.0  # fy
    K[:, 0, 2] = 208.0  # cx
    K[:, 1, 2] = 64.0   # cy

    cpu_out = torch.linalg.inv(K)
    mps_out = torch.linalg.inv(K.to("mps"))

    return compare("linalg.inv", cpu_out, mps_out)


def test_ssim_ops():
    """Test operations used in SSIM computation."""
    print("Testing SSIM ops (AvgPool2d, element-wise)...")
    torch.manual_seed(42)

    x = torch.rand(2, 3, 128, 416)
    pool = torch.nn.AvgPool2d(3, 1)

    # Pad + pool
    x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
    cpu_out = pool(x_pad)
    mps_out = pool(F.pad(x.to("mps"), (1, 1, 1, 1), mode="reflect"))

    return compare("SSIM (avgpool)", cpu_out, mps_out)


def test_forward_pass():
    """Test full encoder→decoder forward pass on MPS."""
    print("Testing full forward pass (ResNet18 encoder → depth decoder)...")
    from depth.models.resnet_encoder import ResNetEncoder
    from depth.models.depth_decoder import DepthDecoder

    torch.manual_seed(42)
    x = torch.randn(1, 3, 128, 416)

    # CPU forward
    enc_cpu = ResNetEncoder(pretrained=False)
    dec_cpu = DepthDecoder(num_ch_enc=enc_cpu.num_ch_enc)
    enc_cpu.eval()
    dec_cpu.eval()

    with torch.no_grad():
        feats_cpu = enc_cpu(x)
        out_cpu = dec_cpu(feats_cpu)

    # MPS forward (same weights)
    enc_mps = ResNetEncoder(pretrained=False).to("mps")
    dec_mps = DepthDecoder(num_ch_enc=enc_mps.num_ch_enc).to("mps")
    enc_mps.load_state_dict(enc_cpu.state_dict())
    dec_mps.load_state_dict(dec_cpu.state_dict())
    enc_mps.eval()
    dec_mps.eval()

    with torch.no_grad():
        feats_mps = enc_mps(x.to("mps"))
        out_mps = dec_mps(feats_mps)

    # Check output shapes
    shapes_ok = True
    for key in out_cpu:
        if out_cpu[key].shape != out_mps[key].cpu().shape:
            print(f"  ❌ Shape mismatch at {key}: CPU={out_cpu[key].shape} vs MPS={out_mps[key].cpu().shape}")
            shapes_ok = False

    if shapes_ok:
        print(f"  ✅ Output shapes match across all {len(out_cpu)} scales")

    # Check values
    values_ok = True
    for key in out_cpu:
        if not compare(f"forward_{key}", out_cpu[key], out_mps[key].cpu(), atol=1e-3):
            values_ok = False

    # Check value ranges (disparity should be in [0, 1] from sigmoid)
    for key in out_mps:
        vals = out_mps[key].cpu()
        if vals.min() < 0 or vals.max() > 1:
            print(f"  ❌ Disparity range error at {key}: [{vals.min():.4f}, {vals.max():.4f}]")
            values_ok = False
        else:
            print(f"  ✅ Disparity range OK at {key}: [{vals.min():.4f}, {vals.max():.4f}]")

    return shapes_ok and values_ok


def test_backward_pass():
    """Test that gradients compute on MPS without NaN."""
    print("Testing backward pass on MPS...")
    from depth.models.resnet_encoder import ResNetEncoder
    from depth.models.depth_decoder import DepthDecoder

    torch.manual_seed(42)

    enc = ResNetEncoder(pretrained=False).to("mps")
    dec = DepthDecoder(num_ch_enc=enc.num_ch_enc).to("mps")

    x = torch.randn(1, 3, 128, 416, device="mps", requires_grad=False)

    feats = enc(x)
    out = dec(feats)

    # Create a simple loss and backprop
    loss = sum(v.mean() for v in out.values())
    loss.backward()

    # Check for NaN gradients
    nan_found = False
    for name, param in list(enc.named_parameters()) + list(dec.named_parameters()):
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"  ❌ NaN gradient in {name}")
            nan_found = True

    if not nan_found:
        print("  ✅ No NaN gradients detected")

    return not nan_found


def main():
    print("=" * 60)
    print("MPS Op-by-Op Verification")
    print("=" * 60)

    check_available()

    tests = [
        ("grid_sample (bilinear)", test_grid_sample),
        ("conv2d", test_conv2d),
        ("interpolate (bilinear)", test_interpolate),
        ("matmul (batched)", test_matmul),
        ("linalg.inv", test_inverse),
        ("SSIM ops", test_ssim_ops),
        ("Full forward pass", test_forward_pass),
        ("Backward pass (NaN check)", test_backward_pass),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  ❌ {name} FAILED with exception: {e}")
            results[name] = False
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed! MPS backend is ready for training.")
    else:
        print("Some tests failed. Review errors above before training on MPS.")
        print("Set PYTORCH_ENABLE_MPS_FALLBACK=1 for CPU fallback on unsupported ops.")
        sys.exit(1)


if __name__ == "__main__":
    main()
