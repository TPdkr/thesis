import torch
print(f"TORCH VERSION: {torch.__version__}")
print(f"XPU DEVICE IS AVAIALABLE: {torch.xpu.is_available()}")
print(f"XPU DEVICE COUNT: {torch.xpu.device_count()}")