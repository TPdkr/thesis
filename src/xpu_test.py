import torch
print(torch.xpu.is_available())
print(torch.xpu.device_count())