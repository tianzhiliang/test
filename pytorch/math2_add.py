from __future__ import print_function
import torch
import math
import numpy

def sum(input, axes, keepdim=False):
    # probably some check for uniqueness of axes
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(ax)
    return input

x = torch.rand(3,2,4)
print("random x: ", x)
print("torch.sum(x, [0])", sum(x, [0]), sum(x, [0]).size())
print("torch.sum(x, [1])", sum(x, [1]), sum(x, [1]).size())
print("torch.sum(x, [0,2])", sum(x, [0,2]), sum(x, [0,2]).size())
print("torch.sum(x, [0])", sum(x, [0], keepdim=True), sum(x, [0], keepdim=True).size())
print("torch.sum(x, [1])", sum(x, [1], keepdim=True), sum(x, [1], keepdim=True).size())
