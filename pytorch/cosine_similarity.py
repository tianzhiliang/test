import torch
import torch.nn.functional as F
import numpy as np

from sklearn import metrics
import sklearn.metrics.pairwise

def cos_v1(a, b):
    for bb in b:
        yield F.cosine_similarity(a, bb)

def cos(a, b):
    print("a:", a)
    print("b:", b)

    if 1 == len(a.shape):
        assert len(a.shape) == len(b.shape)
        a_duplicate = a.repeat(1,1)
        b_duplicate = b.repeat(1,1)
    else:
        vec_dim = a.shape[-1]
        assert b.shape[-1] == vec_dim

        a_size = a.shape[0]
        b_size = b.shape[0]
        a_duplicate = torch.t(a).repeat(b_size,1,1).transpose(1,2).transpose(0,1)
        #print("a_duplicate:", a_duplicate)
        #print("a_duplicate.shape:",a_duplicate.shape)
        b_duplicate = b.repeat(a_size, 1, 1)
        #print("b_duplicate:", b_duplicate)
        #print("b_duplicate.shape:", b_duplicate.shape)
    cos = F.cosine_similarity(a_duplicate, b_duplicate, dim=-1)
    print("cos:", cos)
    return cos

def test_cases_cos():
    a=torch.tensor([[1,2,3.0],[0,0,0.0]])
    b=torch.tensor([[1,2,3.1],[-1,-2,-3.0]])
    a2=torch.tensor([[1,2,3.0],[0,0,0]])
    b2=torch.tensor([[1,2,3.1],[-1,-2,-3.0],[5,5,5.0],[6,6,6.0]])
    a3=torch.tensor([1,2,3.0])
    b3=torch.tensor([1,2,3.1])
    a31=torch.tensor([[1,2,3.0]])
    b31=torch.tensor([[1,2,3.1]])

    ar=np.random.rand(5,10)
    br=np.random.rand(15,10)
    art=torch.tensor(ar)
    brt=torch.tensor(br)

    cos(a,b)
    cos(a2,b2)
    abrt = cos(art,brt)
    print("sklearn cos:", sklearn.metrics.pairwise.cosine_similarity(ar,br))
    cos(a3,b3)
    try:
        cos(a3,b3)
    except:
        print("cos(a3,b3) failed")
    try:
        cos([a3],[b3])
    except:
        print("cos(a3,b3) failed")
    cos(a31,b31)

    
test_cases_cos()
