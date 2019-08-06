from __future__ import print_function
import torch

print("dimshuffle:")
a = torch.rand(1,2,3,4)
#print("random x: ", a)
print("random x: ", a.size())
#print("dimshuffle x by transpose: ", a.transpose(0,3).transpose(1,2))
print("dimshuffle x by transpose: ", a.transpose(0,3).transpose(1,2).size())
#print("dimshuffle x by permute: ", a.permute(3,2,1,0))
print("dimshuffle x by permute: ", a.permute(3,2,1,0).size())
print("dimshuffle x by transpose: ", a.transpose(0,1).size())

a = torch.rand(1,2,3)
print("random x: ", a.size())
print("dimshuffle x by permute: ", a.permute(1,0,2).size())
print("dimshuffle x by transpose: ", a.transpose(0,1).size())
