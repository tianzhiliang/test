import torch
import torch.nn as nn

dim1=3
dim2=5
seqlen1=6
bs=10
seqlen2=7
x=torch.rand([bs,seqlen1,dim1])
y=torch.rand([bs,seqlen2,dim2])
proj=nn.Linear(dim1, dim2)

xp = proj(x)
yt = y.transpose(2,1)
res = xp.bmm(yt)

print("x.shape", x.shape)
print("y.shape", y.shape)
print("xp.shape:", xp.shape)
print("yt.shape:", yt.shape)
print("res.shape:", res.shape)
