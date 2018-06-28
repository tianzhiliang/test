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

mean = torch.zeros(3,2)
variance_square = torch.ones(3,2)
print("mean(3*2): ",mean)
print("variance_square(3*2): ", variance_square)
print("normal distribution(3*2): ")
for i in range(5):
    print(torch.normal(mean, variance_square), " ")

mean1 = torch.zeros(mean.size())
variance_square1 = torch.ones(variance_square.size())
epsilon = torch.normal(mean1, variance_square1)
a = variance_square / 2
b = torch.exp(a)
c = b * epsilon
print("c:", c)
