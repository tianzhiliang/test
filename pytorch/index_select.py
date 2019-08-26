import torch
 
 
x = torch.linspace(1, 12, steps=12).view(3,4)
     
print(x)
indices = torch.LongTensor([0, 2])
y = torch.index_select(x, 0, indices)
print(y)
 
z = torch.index_select(x, 1, indices)
print(z)
 
z = torch.index_select(y, 1, indices)
print(z)
