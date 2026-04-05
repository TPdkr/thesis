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
        device = "cuda" #NVIDIA
    elif torch.backends.mps.is_available():
        device = "mps" #Apple🍏
    elif torch.xpu.is_available():
        device = "xpu" #Intel
    else:
        device = "cpu" #Default

    print(f"Using device: {device} :3")
    return device

getDevice()