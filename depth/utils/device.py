"""Device selection utility with MPS/CUDA/CPU fallback."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Select the best available compute device.

    Args:
        preference: One of "auto", "mps", "cuda", "cpu".
            "auto" picks CUDA > MPS > CPU in priority order.

    Returns:
        torch.device for the selected backend.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS requested but not available. "
                "Requires macOS 12.3+ and Apple Silicon."
            )
        return torch.device("mps")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def device_summary(device: torch.device) -> str:
    """Return a human-readable summary of the device."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        mem_gb = torch.cuda.get_device_properties(device).total_mem / (1024**3)
        return f"CUDA: {name} ({mem_gb:.1f} GB)"
    elif device.type == "mps":
        return "MPS: Apple Silicon GPU (unified memory)"
    else:
        return "CPU"
