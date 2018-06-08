from __future__ import print_function
import torch

x = torch.empty(5, 3) # invalid for pytorch 0.3.0
print(x)

x = torch.rand(5, 3)
print("random:", x)

x = torch.zeros(5, 3, dtype=torch.long) # there is no "long" in 
print("all zero:", x)

x = torch.zeros(5, 3, dtype=torch.float)
print("all zero:", x)

y = torch.rand(5, 3)
print("zeros + randoms", x + y)

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print("add on gpu:", z)
    print("add on cpu", z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
