from __future__ import print_function
import torch

x = torch.rand(2, 3, 4, 2)
print("x:", x)
print("cat(x,dim=0):", torch.cat(x, dim=0))
print("cat(x,dim=1):", torch.cat(x, dim=1))
print("cat(x,dim=2):", torch.cat(x, dim=2))
