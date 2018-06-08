from __future__ import print_function
import torch
from torch.autograd import Variable

#mean = torch.FloatTensor(0)
#variance_square = torch.FloatTensor(1)

for i in range(5):
    print(torch.normal(torch.tensor([0.0]), torch.tensor([1.0])), " ")
print("")
mu = Variable(torch.Tensor([1]), requires_grad=True)
sigma = Variable(torch.Tensor([1]), requires_grad=True)
for i in range(5):
    print(torch.normal(mu, sigma))
print("")
mean = torch.zeros(3)
variance_square = torch.ones(3)
print("mean: ",mean)
print("variance_square: ", variance_square)
for i in range(5):
    print(torch.normal(mean, variance_square), " ")
