from __future__ import print_function
import torch
import math
import numpy

print("torch.exp(torch.rand(2))", torch.exp(torch.rand(2)))
print("torch.log(torch.exp(torch.rand(2)))", torch.log(torch.exp(torch.rand(2))))
#print("torch.square(torch.rand(2))", torch.square(torch.rand(2)))
#print("torch.exp(4):", torch.exp(4)) # can not work
print("math.exp(4):", math.exp(4))
#print("math.exp(torch.rand(2)):",math.exp(torch.rand(2))) # can not work
print("numpy.random.rand(3,2)", numpy.random.rand(3,2))
x = torch.rand(3,2)
print("random x: ", x)
print("torch.sum(x)", torch.sum(x))
#print("torch.sum(torch.rand(2))", torch.sum(torch.rand(3,2), axis=1))
#print("keras.square():", keras.square(numpy.random.rand(3,2)))

print("x + 1 (\"naive +\" means add an identity matrix): ", x + 1)
print("x + 3 (\"naive +\" means add an identity matrix): ", x + 3)

y = torch.rand(3,2)
print("random y: ", y)
tmp1 = y - x
print("y - x:", tmp1)
print("y - x:", y - x)
