#import torch
from torch import cuda

#print("cuda is_available:", torch.cuda.is_available())
cuda.set_device(1)
#cuda.set_device(1)
#cuda.set_device(2)
print("done")
