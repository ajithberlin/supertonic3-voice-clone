import torch


def get_device(allow_mps: bool = True) -> str:
    """Picks the fastest available torch device: CUDA > MPS (Apple GPU) > CPU.

    Args:
        allow_mps: Set False for code paths (e.g. speechbrain's EncoderClassifier)
            that don't support the "mps" device string and would crash on it.
    """
    if torch.cuda.is_available():
        return "cuda:0"
    if allow_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
