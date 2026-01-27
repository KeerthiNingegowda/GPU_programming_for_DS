import torch


x = torch.randn(1000, 1000, device="cuda")
y = torch.randn(1000,1000, device="cuda")

z = x @ y
print(z.device)
print(z[0])



