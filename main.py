import torch

print(torch.cuda.is_available())

print(torch.cuda.device_count())

if torch.cuda.is_available():
    print(f"GPU name {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")


