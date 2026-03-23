import torch

def getDevice():
    """
    Get the best available device for pytorch. If a GPU is
    available GPU type is returned. Otherwise CPU is returned.

    Returns:
        String: device type available
    """
    device=None
    if torch.cuda.is_available():
        device = "cuda" # Use NVIDIA GPU (if available)
    elif torch.backends.mps.is_available():
        device = "mps" # Use Apple Silicon GPU (if available)
    elif torch.xpu.is_available():
        device = "xpu" # Use intel GPU
    else:
        device = "cpu" # Default to CPU if no GPU is available
    return device