from __future__ import print_function
import torch

x = torch.randn(4, 4)
print("4*4:", x.size())
y = x.view(16)
print("(4*4)->(16):", y.size())
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print("(4*4)->(-1*8)", z.size())
print("\n")
x = torch.randn(1, 2, 3)
print("(1*2*3):", x.size())
print("(1*2*3)->(2*3) means eliminate (1):", x.view(x.size()[1:]).size())
