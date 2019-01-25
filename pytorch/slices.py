from __future__ import print_function
import torch

x = torch.rand(2, 3, 4)
a=(x,x)
print("random x:", x)
print("a=(x,x):", a)
print("len a:", len(a))
print("a[]:", a[:])
print("a[]:", a[:,1,:])
print("x size:", x.size())
print("x[] size:", x[:,1,:])
print("x[] size:", x[:,[1,2],:])
