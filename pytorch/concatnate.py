from __future__ import print_function
import torch

x = torch.rand(2, 3, 4, 2)
print("x:", x.size())
print("cat([x[0], x[1]],dim=0):", torch.cat([x[0], x[1]], dim=0).size())
print("cat([x[0], x[1]],dim=1):", torch.cat([x[0], x[1]], dim=1).size())
print("cat([x[0], x[1]],dim=2):", torch.cat([x[0], x[1]], dim=2).size())

print("x:", x)
print("cat([x[0], x[1]],dim=0):", torch.cat([x[0], x[1]], dim=0))
print("cat([x[0], x[1]],dim=1):", torch.cat([x[0], x[1]], dim=1))
print("cat([x[0], x[1]],dim=2):", torch.cat([x[0], x[1]], dim=2))
