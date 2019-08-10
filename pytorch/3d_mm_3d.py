import torch

# from https://discuss.pytorch.org/t/batch-3dtensor-matrix-multiplication/12092
len1=6
len2=4
batchsize=3
embdim=5
b=torch.rand([len1,batchsize,embdim])
e=torch.rand([len2,batchsize,embdim])

print("mm 1: output: (len1, len2)")
print("b.view(6,15).shape:", b.view(len1,batchsize*embdim).shape)
print("e.view(4,15).t().shape:", e.view(len2,batchsize*embdim).t().shape)
print("b.view(6,15).mm(e.view(4,15).t()).shape:", b.view(len1,batchsize*embdim).mm(e.view(len2,batchsize*embdim).t()).shape)

print("mm 2: output: (len1*batchsize, len2*batchsize)")
print("b.view(18,5).shape:", b.view(len1*batchsize,embdim).shape)
print("e.view(12,5).shape:", e.view(len2*batchsize,embdim).shape)
print("b.view(18,5).mm(e.view(12,5).t()).shape:", b.view(len1*batchsize,embdim).mm(e.view(len2*batchsize,embdim).t()).shape)

print("batch mm 3: output: (batchsize, len1, len2)")
m1=torch.rand([batchsize,len1,embdim])
m2=torch.rand([batchsize,embdim,len2])
print("input m1 ([batchsize,len1,embdim]):", m1.shape)
print("input m2 ([batchsize,embdim,len2]):", m2.shape)
print("output torch.bmm(m1,m2) ([batchsize,len1,len2]):", torch.bmm(m1,m2).shape)
print("m1:",m1)
print("m2:",m2)
print("torch.bmm(m1,m2):",torch.bmm(m1,m2))
