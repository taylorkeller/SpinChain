import torch
print(torch.cuda.is_available())  # Should print True for GPU
print(torch.cuda.get_device_name(0))  # Should print your GPU
