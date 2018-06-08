from __future__ import print_function
import torch

# toturial https://stackoverflow.com/questions/44524901/how-to-do-product-of-matrices-in-pytorch

a = torch.rand(2, 3)
b = torch.rand(3, 4)
c = torch.ones(3, 4)
cv = torch.ones(3)
print("random a:", a)
print("random b:", b)
print("all ones c:", c)
print("all ones cv:", cv)

print("matrix multiplication by torch.mm(a, b): ", torch.mm(a, b))
print("matrix multiplication by torch.mm(a, c): ", torch.mm(a, c))
print("matrix multiple vevtor by torch.mv(a, cv): ", torch.mv(a, cv))
print("matrix multiplication by a.mm(b): ", a.mm(b))
print("matrix multiplication by a @ b: ", a @ b)

d = torch.ones(2, 3)
print("element-wise multiplication")
try:
    print("element-wise multiplication by a * b: ", a * b)
except:
    print("element-wise multiplication by a * b (size error) ")
print("element-wise multiplication by a * d: ", a * d)
print("element-wise multiplication by mul(a, d): ", torch.mul(a, d))
 
print("dot product")
# a.dot(d) # size error
try:
    aa = torch.rand(3, 1)
    bb = torch.ones(3, 1)
    aa.dot(bb)
except:
    print("dot product by 3*1 dot 3*1: size error")
aa = torch.rand(3)
bb = torch.ones(3)
print("aa:", aa)
print("bb:", bb)
print("dot product by aa*bb (3*3) means dot product:", aa.dot(bb))

print("number multiples matrix/tensor")
print("number*tensor 5 * a: ", 5 * a)
