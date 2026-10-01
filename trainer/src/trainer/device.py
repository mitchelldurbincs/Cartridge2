"""Device selection shared by native and container training."""


def resolve_device(requested: str) -> str:
    import torch

    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if requested == "cuda" and not cuda:
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch build, "
            "a compatible NVIDIA driver and container GPU passthrough; CPU fallback is disabled."
        )
    if requested == "mps" and not mps:
        raise RuntimeError(
            "MPS was requested but is unavailable on this host; CPU fallback is disabled."
        )
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown training device: {requested}")
    return requested
