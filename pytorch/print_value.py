from __future__ import print_function
import torch

x = torch.rand(2, 3)
print("random x:", x)
print("x[0][0]:", x[0][0])
#print("x.data.cpu().numpy()[0]:", x.data.cpu().numpy()[0])
print("str x[0].value:", str(x[0][0]))
print("str x[0].value:", " ".join(list(map(str, x[0]))))
#print("x[0].value:", x[0][0].data[0])

#y = torch.rand(2, 3).cuda()
#print("random y:", y)
#print("y[0][0]:", y[0][0])
#print("y[0].value:", y[0][0].value)
