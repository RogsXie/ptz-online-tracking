"""Explicit compute selection shared by the CPU and CUDA entry points."""
import sys
import torch


def resolve_device(requested="cpu"):
    requested = str(requested)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA requested but unavailable in {sys.executable} "
            f"(torch={torch.__version__}, CUDA build={torch.version.cuda}). "
            "Use the 50mamba interpreter for GPU runs."
        )
    return str(device)


def compute_metadata(device):
    import torchvision
    result = {"device": str(device), "python": sys.executable,
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "cuda_build": torch.version.cuda}
    if torch.device(device).type == "cuda":
        result.update(gpu=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)))
    return result
