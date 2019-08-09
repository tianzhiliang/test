import torch

# from https://discuss.pytorch.org/t/batch-3dtensor-matrix-multiplication/12092
b=torch.rand([6,3,5])
e=torch.rand([4,3,5])

print("mm 1")
print("b.view(6,15).shape:", b.view(6,15).shape)
print("e.view(4,15).t().shape:", e.view(4,15).t().shape)
print("b.view(6,15).mm(e.view(4,15).t()).shape:", b.view(6,15).mm(e.view(4,15).t()).shape)

print("mm 2")
print("b.view(18,5).shape:", b.view(18,5).shape)
print("e.view(12,5).shape:", e.view(12,5).shape)
print("b.view(18,5).mm(e.view(12,5).t()).shape:", b.view(18,5).mm(e.view(12,5).t()).shape)
