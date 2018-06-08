from __future__ import print_function
import torch

x = torch.rand(5, 3)
print("random:", x)
print("shape x:", x.size())
print("ones follow shape of x:", torch.ones(x.size()))
