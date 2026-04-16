import torch
import random
import numpy as np

#DEVICE ACCESS :3==============================================================
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

    return device

likely_device = getDevice()
print(f"Using device: {likely_device} :3")

#REPRODUCIBILITY :3============================================================
SEED = 42

def setSeed(seed=SEED):
    """
    Set the seed for everyhting to be the same to ensure reproducibility.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if getDevice()=="xpu":
        torch.xpu.manual_seed(seed)

print(f"Setting seed to {SEED}:3")
setSeed()

#TRAINING FUNCTION FOR PREPROCESSING :3========================================

def collate_fn(batch):
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets