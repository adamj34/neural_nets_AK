import torch

t = torch.zeros(3,3, dtype=torch.int32)
# print(t)
t[0,1] = 43
t=t[1:,]
print(t)