import torch
from torch.autograd import Variable

a=torch.tensor([1.1,2,3])
b=torch.tensor([[-1,-2.1,3.1],[0.5,3,3]])
c=torch.tensor([[1.1,2,3],[4,5,6],[7,8,9],[10,11,12]])

av=Variable(a, requires_grad=True)
cv=Variable(c, requires_grad=True)
bnp=b.numpy()

try:
    av*bnp
except:
    print("a(Variable)*bnp(numpy) failed!")

print("a.requires_grad:", a.requires_grad)
print("av.requires_grad:", av.requires_grad)
print("av*b.requires_grad:", (av*b).requires_grad)

print("av:", av)
print("b:", b)
print("c:", c)
print("av*b:", av*b)
print("b*av:", b*av)
print("av*c:", av*c)
print("av*cv:", av*cv)
print("cv*av:", cv*av)
print("torch.sum(av*b, dim=0):", torch.sum(av*b, dim=0))
print("torch.sum(av*b, dim=1):", torch.sum(av*b, dim=1))
print("torch.sum(av*cv, dim=0):", torch.sum(av*cv, dim=0))
print("torch.sum(av*cv, dim=1):", torch.sum(av*cv, dim=1))
print("torch.sum(av*c, dim=0):", torch.sum(av*c, dim=0))
print("torch.sum(av*c, dim=1):", torch.sum(av*c, dim=1))
try:
    print("cv*b:", cv*b)
    print("b*cv:", b*cv)
    print("torch.sum(cv*b, dim=0):", torch.sum(cv*b, dim=0))
    print("torch.sum(cv*b, dim=1):", torch.sum(cv*b, dim=1))
except:
    print("cv*b failed")
#cv_res = Variable(torch.tensor([]), requires_grad=True)
for ccv in cv:
    print("ccv*b:", ccv*b)
#    cv_res += ccv*b
print("cv_by_list:", [ccv*b for ccv in cv])
print("\n")

# Matrix(dim2, dim1) dot_product with BatchVector(batchsize, dim1)
batchsize=3
dim1=2
dim2=4
v = torch.randn(batchsize, dim1)
M = torch.randn(dim2, dim1)
print("v:",v)
print("torch.t(v):",torch.t(v))
print("M:",M)
print("M*v^T:", M.matmul(torch.t(v)))
print("\n")

# BatchMatrix(batch_size, dim2, dim1) element wise product with BatchVector(batchsize, dim2)
dim1=2
dim2=3
batch_size=4
x=torch.rand(dim2, dim1)
x_batch=torch.rand(batch_size, dim2, dim1)
y=torch.rand(batch_size, dim2)
print("x.shape:",x.shape)
print("x_batch.shape:",x_batch.shape)
print("y.shape:",y.shape)
#print("x:",x)
#print("x_batch:",x_batch)
#print("y:",y)

try:
    print("x_batch*y:",x_batch*y)
except:
    print("x_batch*y failed:")

x_batch=torch.rand(dim1, dim2, batch_size)
y=torch.rand(dim2, batch_size)
print("x_batch.shape:",x_batch.shape)
print("y.shape:",y.shape)
print("x_batch*y.shape:",(x_batch*y).shape)
print("x_batch:",x_batch)
print("y:",y)
print("x_batch*y:",x_batch*y)
print("\n")

# Matrix(dim1, dim2) element wise product with BatchVector(batchsize, dim1) get (batchsize, dim1, dim2) 
M=torch.rand(dim1, dim2)
v=torch.rand(batch_size, dim1)
M_batch = M.repeat(batch_size, 1, 1)
print("M.shape:",M.shape)
print("v.shape:",v.shape)
print("M_batch.shape:",M_batch.shape)
M_batch_transpose = M_batch.transpose(0,2)
v_transpose = v.transpose(0,1)
print("M_batch_transpose.shape:",M_batch_transpose.shape)
print("v_transpose.shape:",v_transpose.shape)
#print("M_batch_transpose:",M_batch_transpose)
#print("v_transpose:",v_transpose)
Mv = M_batch_transpose * v_transpose
#print("M_batch_transpose * v_transpose:", Mv)
print("M_batch_transpose * v_transpose shape:", Mv.shape)
final = Mv.transpose(0,2)
print("Input: M:", M)
print("Input: v:", v)
print("Output:", final)
print("\n")

# Matrix(dim1, dim2) weighted sum up by BatchVector(batchsize, dim1) get (batchsize, dim2) 
M=torch.rand(dim1, dim2)
v=torch.rand(batch_size, dim1)
M_batch = M.repeat(batch_size, 1, 1)
M_batch_transpose = M_batch.transpose(0,2)
v_transpose = v.transpose(0,1)
Mv = M_batch_transpose * v_transpose
Mv_transpose = Mv.transpose(0,2)
print("M_batch_transpose * v_transpose transpose shape:", Mv_transpose.shape)
Mv_transpose_sum = torch.sum(Mv_transpose, dim=1)
print("Mv_transpose_sum.shape:",Mv_transpose_sum.shape)
print("Input: M:", M)
print("Input: v:", v)
print("Output:", Mv_transpose_sum)
print("\n")
