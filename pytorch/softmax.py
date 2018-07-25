from __future__ import print_function
import torch

#m = torch.nn.Softmax()
#input = torch.Variable(torch.randn(2, 3))
#print(input)
#print(m(input))

x = torch.rand(2, 3)
b = torch.rand(2)
print("random x:", x)
#print("softmax(x):", torch.nn.functional.softmax(x, 0))
print("softmax(x):", torch.nn.functional.softmax(b))
