import torch

batchsize=3
classnum=6
seqlen=5

loss_normal=torch.nn.CrossEntropyLoss()
loss_unfold=torch.nn.CrossEntropyLoss(reduction="none")

target=torch.randint(0,classnum-1,[batchsize])
output=torch.rand([batchsize,classnum])

print("target.shape:", target.shape)
print("output.shape:", output.shape)
print("loss_normal(output,target):", loss_normal(output,target))

target=torch.randint(0,classnum-1,[batchsize,seqlen])
output=torch.rand([batchsize,classnum,seqlen])
print("target.shape:", target.shape)
print("output.shape:", output.shape)
print("loss_normal(output,target):", loss_normal(output,target))
print("loss_unfold(output,target).shape:", loss_unfold(output,target).shape)
print("loss_unfold(output,target):", loss_unfold(output,target))
