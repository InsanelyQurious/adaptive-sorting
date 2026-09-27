"""CUDA detection (Section 29). Never raises when CUDA is absent."""
from __future__ import annotations
import torch


def detect_device(preference: str = "auto") -> tuple[str, str | None]:
    if preference not in ("auto", "cuda") or not torch.cuda.is_available():
        return "cpu", None
    try:
        return "cuda", torch.cuda.get_device_name(0)
    except Exception:
        return "cpu", None


def device_label(dev: str, name: str | None) -> str:
    return f"CUDA — {name}" if dev == "cuda" and name else ("CUDA" if dev == "cuda" else "CPU")


def gpu_utilization() -> float | None:
    """Return GPU utilization in percent if a CUDA device with NVML is present, else None."""
    if not torch.cuda.is_available():
        return None
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
    except Exception:
        try:
            return float(torch.cuda.utilization(0))
        except Exception:
            return None
