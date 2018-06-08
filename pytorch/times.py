from __future__ import print_function
import torch

x = torch.rand(5, 3)
print("random x:", x)

y = torch.rand(5, 3) 
print("random y:", y)

z = torch.ones(5, 3) 
print("all one z:", z)

k = torch.tensor([5])
print("single float k: ", k)

print("x * y:", x * y)
print("x * z:", x * z)
print("x * 5:", x * 5)
print("x * k:", x * k)
