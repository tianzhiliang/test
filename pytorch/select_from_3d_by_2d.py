import torch

batchsize=3
embdim=5
a=torch.tensor([[1, 2, 3], [4, 5, 0]])
b=torch.rand([6,batchsize,embdim])
c=b.reshape(6*batchsize,embdim)
e=torch.rand([4,batchsize,embdim])

print("b[a].shape:", b[a].shape)
indx_for_2dt = torch.tensor([[batch_id + batchsize * idx for idx in idxs] for batch_id, idxs in enumerate(a)])
print("indx_for_2dt.shape:", indx_for_2dt.shape)
print("c[indx_for_2dt].shape:", c[indx_for_2dt].shape)
print("b",b)
print("c:", c)
print("b[a]:",b[a])
print("indx_for_2dt:", indx_for_2dt)
print("c[indx_for_2dt]:", c[indx_for_2dt])
