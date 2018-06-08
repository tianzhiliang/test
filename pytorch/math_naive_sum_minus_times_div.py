from __future__ import print_function
import torch
import math
import numpy

x = torch.rand(3,2)
y = torch.rand(3,2)
print("random x: ", x)
print("random y: ", y)
print("x + 3 (\"naive +\" means add tensor_with_a_same_number):", x + 3)
print("x + y (\"naive +\" means element-wise add two tensor:", x + y)

print("x - 5 means tensor - tensor_with_a_same_number:", x - 5)
print("x - y means element-wise minus:", x - y)

print("5 * x means number * tensor: ", 5 * x)
print("x * y menas element-wise multiple ", x * y)

print("x / 5 means tensor / number:", x / 5)
print("x / y:", x / y)

#a = torch.rand(4, 3)
#b = torch.rand(4)
#print("a:", a)
#print("b:", b)
#print("a / 5 means tensor / number:", a / 5)
#print("a / b:", a/b)
