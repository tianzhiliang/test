from __future__ import print_function
import torch

x = torch.randn(4, 4)
print(x.size())
y = x.view(16)
print(y.size())
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(z.size())
