from __future__ import print_function
import torch

#m = torch.nn.Softmax()
#input = torch.Variable(torch.randn(2, 3))
#print(input)
#print(m(input))

x = torch.rand(2, 3)
b = torch.rand(2)
print("random x:", x)
print("random x.shape:", x.shape)
print("softmax(x,dim=0):", torch.nn.functional.softmax(x, dim=0))
print("softmax(x,dim=1):", torch.nn.functional.softmax(x, dim=1))
print("softmax(b):", torch.nn.functional.softmax(b, dim=0))
