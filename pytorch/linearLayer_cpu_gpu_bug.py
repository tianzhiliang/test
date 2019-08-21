import torch
import torch.nn as nn

a=torch.rand([3,2,4],device=torch.device("cuda:1"))
b=torch.rand([2,4],device=torch.device("cuda:1"))
d=torch.cat(list(a)+[b],dim=0)

# normal coding: will cause error
fc1=nn.Linear(4,3)
try:
    fc1(a)
    fc1(b)
    fc1(d)
except Exception as e:
    print("run fc1 (nn.Linear(4,3)) error")
    print("error:", e)

# correct coding: both of them use cuda
fc1.to(torch.device("cuda:1"))
print("fc1(a).shape:", fc1(a).shape)
print("fc1(b).shape:", fc1(b).shape)
print("fc1(d).shape:", fc1(d).shape)
