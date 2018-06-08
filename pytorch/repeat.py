from __future__ import print_function
import torch

x = torch.rand(2, 3)
print("random x:", x)
print("x repeat(4,2) :", x.repeat(4, 2))
print("size x:", x.size())
print("size x repeat 4,2:", x.repeat(4, 2).size())
print("size x repeat 4,2,3:", x.repeat(4, 2, 3).size())

print("repeat along axis (assigning axis)")
print("size x repeat 10,1,1:", x.repeat(10, 1, 1).size())
print("size x repeat 10,1:", x.repeat(10, 1).size())
print("size x repeat 1,10:", x.repeat(1, 10).size())
print("size x repeat 1,10:", x.repeat(1, 10).size())
print("size x repeat 10,1,1 then permute:", x.repeat(10, 1, 1).permute(1,2,0).size())
print("x: ", x)
print("size x repeat 10,1,1 then permute:", x.repeat(10, 1, 1).permute(1,2,0))

print("repeat from 1d to 3d")
a = torch.rand(3)
print("random a:", a)
print("random a size:", a.size())
print("size a repeat 10, 1 then repeat 10, 1", a.repeat(10, 1).repeat(10, 1, 1).size())
