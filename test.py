import torch

a = torch.randn(4, 4)
# print(a)

# print(torch.max(a, 1))

print(a.max(1, keepdim=True))