import torch

batchsize=3
embdim=5
a=torch.tensor([[1, 2, 3], [4, 5, 0]])
b=torch.rand([6,batchsize,embdim])
c=b.reshape(6*batchsize,embdim)
e=torch.rand([4,batchsize,embdim])

print("b[a].shape:", b[a].shape)
a_t = a.t()
indx_for_2dt = torch.tensor([[batch_id + batchsize * idx for idx in idxs] for batch_id, idxs in enumerate(a_t)])
print("a_t.shape:", a_t.shape)
print("indx_for_2dt.shape:", indx_for_2dt.shape)
print("c[indx_for_2dt].transpose(0, 1).shape (final result):", c[indx_for_2dt].transpose(0, 1).shape)
print("b",b)
print("c:", c)
print("b[a]:",b[a])
print("indx_for_2dt:", indx_for_2dt)
print("c[indx_for_2dt]:", c[indx_for_2dt])
print("c[indx_for_2dt].transpose(0, 1) (final result):", c[indx_for_2dt].transpose(0, 1))
#print("c[indx_for_2dt].transpose(1, 0) (final result):", c[indx_for_2dt].transpose(1, 0))
