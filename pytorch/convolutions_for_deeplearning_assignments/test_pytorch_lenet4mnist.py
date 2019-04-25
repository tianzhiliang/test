import torch
import torch.nn as nn

# mnist for LeNet

conv1=nn.Conv2d(1, 6, (1, 1), stride=1)
p1=nn.AvgPool2d((2, 2), stride=2)
conv2=nn.Conv2d(6,16,(5,5),stride=1)
p2=nn.AvgPool2d((2, 2), stride=2)
conv3=nn.Conv2d(16,120,(5,5),stride=1)
fc1 = nn.Linear(120, 84)
fc2 = nn.Linear(84, 10)

mnist_input = torch.randn(128,1,28,28)
print("input:", mnist_input.size())
print("conv1 result:", conv1(mnist_input).size())
print("p1 result:", p1(conv1(mnist_input)).size())
print("conv2 result:", conv2(p1(conv1(mnist_input))).size())
print("p2 result:", p2(conv2(p1(conv1(mnist_input)))).size())
print("conv3 result:", conv3(p2(conv2(p1(conv1(mnist_input))))).size())
x = conv3(p2(conv2(p1(conv1(mnist_input)))))
print("reshape result:", x.view(x.shape[:2]).size())
print("fc1 result:", fc1(x.view(x.shape[:2])).size())
print("fc2 result:", fc2(fc1(x.view(x.shape[:2]))).size())
