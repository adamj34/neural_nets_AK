import torch

g = torch.Generator().manual_seed(2147483647)
t = torch.randn((3,4), generator=g) * 3
b1 = torch.randn(4, generator=g)
print(t)
print(b1)
print(t + b1)