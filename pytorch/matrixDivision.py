from __future__ import print_function
import torch

a = torch.rand(2, 3)
b = torch.rand(2, 3)
print("a:", a)
print("b:", b)
print("div(a,b) means element wise division:", torch.div(a, b))
print("a / b means element wise division:", a / b)
