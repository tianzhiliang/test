from torch.nn.parameter import Parameter
#from __future__ import print_function
import torch

row = 10
col = 15
a = Parameter(torch.Tensor(row, col))
torch.nn.init.constant_(a, 1)
print(a)
