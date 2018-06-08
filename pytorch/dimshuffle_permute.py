from __future__ import print_function
import torch

print("dimshuffle:")
a = torch.rand(1,2,3,4)
print("random x: ", a)
print("dimshuffle x by transpose: ", a.transpose(0,3).transpose(1,2))
print("dimshuffle x by transpose: ", a.transpose(0,3).transpose(1,2).size())
print("dimshuffle x by permute: ", a.permute(3,2,1,0))
print("dimshuffle x by permute: ", a.permute(3,2,1,0).size())
