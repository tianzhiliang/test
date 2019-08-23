import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

batchsize=3
classnum=6
seqlen=5
dim1=classnum

class Generator(nn.Module):
    def __init__(self, batchsize, classnum, seqlen, dim1):
        super(Generator, self).__init__()
        self.batchsize=batchsize
        self.classnum=classnum
        self.seqlen=seqlen
        self.dim1=dim1
        self.fc = nn.Linear(self.dim1, self.classnum)
        print("self.fc.weight:", self.fc.weight)
        print("self.fc.bias:", self.fc.bias)
        self.fc.weight = Parameter(torch.tensor([[ 0.1490, -0.0909,  0.0056,  0.2489,  0.2706,  0.3366], \
                [ 0.2262,  0.3199, -0.1338,  0.1753, -0.3995, -0.2002], \
                [ 0.3979,  0.0298,  0.3689, -0.2320, -0.0523, -0.0655], \
                [-0.3470, -0.2024,  0.0705,  0.1303, -0.0123, -0.2240], \
                [-0.0320, -0.0361,  0.0675,  0.3214,  0.3430, -0.3797], \
                [ 0.2855, -0.2217,  0.1066,  0.0015, -0.1856, -0.1491]], requires_grad=True))
        self.fc.bias = Parameter(torch.tensor([ 0.3170,  0.2004, -0.3995, -0.2146,  0.2227,  0.3299], \
                    requires_grad=True))
        print("self.fc.weight:", self.fc.weight)
        print("self.fc.bias:", self.fc.bias)

    def forward(self, input):
        output = self.fc(input)
        return output

class GeneratorMasked(nn.Module):
    def __init__(self, batchsize, classnum, seqlen, dim1):
        super(GeneratorMasked, self).__init__()
        self.batchsize=batchsize
        self.classnum=classnum
        self.seqlen=seqlen
        self.dim1=dim1
        self.fc = nn.Linear(self.dim1, self.classnum)
        #print("self.fc.parameters():", self.fc.parameters())
        print("self.fc.weight:", self.fc.weight)
        print("self.fc.bias:", self.fc.bias)
        self.fc.weight = Parameter(torch.tensor([[ 0.1490, -0.0909,  0.0056,  0.2489,  0.2706,  0.3366], \
                [ 0.2262,  0.3199, -0.1338,  0.1753, -0.3995, -0.2002], \
                [ 0.3979,  0.0298,  0.3689, -0.2320, -0.0523, -0.0655], \
                [-0.3470, -0.2024,  0.0705,  0.1303, -0.0123, -0.2240], \
                [-0.0320, -0.0361,  0.0675,  0.3214,  0.3430, -0.3797], \
                [ 0.2855, -0.2217,  0.1066,  0.0015, -0.1856, -0.1491]], requires_grad=True))
        self.fc.bias = Parameter(torch.tensor([ 0.3170,  0.2004, -0.3995, -0.2146,  0.2227,  0.3299], \
                    requires_grad=True))
        print("self.fc.weight:", self.fc.weight)
        print("self.fc.bias:", self.fc.bias)

    def forward(self, input, mask):
        output = self.fc(input)
        #output_masked = output * mask.float()
        output_masked = torch.mul(output, mask.float())
        return output_masked

generator = Generator(batchsize, classnum, seqlen, dim1)
masked_generator = GeneratorMasked(batchsize, classnum, seqlen, dim1)
loss_normal=torch.nn.CrossEntropyLoss()

# Bulid net done. will train
#target=torch.randint(0,classnum-1,[batchsize,seqlen])
#input=torch.rand([batchsize,dim1,seqlen], requires_grad = True)
#target=torch.randint(0,classnum-1,[batchsize])
target=torch.tensor([1,2,3])
mask0=torch.tensor([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])
mask1=torch.tensor([[1,1,0,0,0,0],[0,0,1,0,0,0],[1,1,1,1,0,0]])
ninf=-float('inf')
mask2=torch.tensor([[1,1,ninf,ninf,ninf,ninf],[ninf,ninf,1,ninf,ninf,ninf],[1,1,1,1,ninf,ninf]])
nlarge=-1000
mask3=torch.tensor([[1,1,nlarge,nlarge,nlarge,nlarge],[nlarge,nlarge,1,nlarge,nlarge,nlarge],[1,1,1,1,nlarge,nlarge]])
nlarge=-9
mask4=torch.tensor([[1,1,nlarge,nlarge,nlarge,nlarge],[nlarge,nlarge,1,nlarge,nlarge,nlarge],[1,1,1,1,nlarge,nlarge]])
nlarge=-1
mask5=torch.tensor([[1,1,nlarge,nlarge,nlarge,nlarge],[nlarge,nlarge,1,nlarge,nlarge,nlarge],[1,1,1,1,nlarge,nlarge]])
#input=torch.rand([batchsize,dim1], requires_grad = True)
input=torch.tensor([[0.1334, 0.4332, 0.9542, 0.0585, 0.7362, 0.9372],[0.1247, 0.2000, 0.6954, 0.4101, 0.0087, 0.5814],[0.5126, 0.5473, 0.4716, 0.9520, 0.7300, 0.0541]], requires_grad = True)
training_steps = 100

def train_cross_entropy():
    for i in range(training_steps):
        cur_generator = generator
        #cur_generator = masked_generator
        #output=cur_generator(input)
        #output=cur_generator(input, mask0)
        #output=cur_generator(input, mask1)
        #output=cur_generator(input, mask2)
        #output=cur_generator(input, mask3)
        print("target.shape:", target.shape)
        print("output.shape:", output.shape)
        print("input.shape:", input.shape)
        print("target:", target)
        print("input:", input)
        print("output:", output)

        #loss_value = loss_normal(output,target)
        loss_value = F.nll_loss(output,target)
        print("loss_normal(output,target):", loss_value)

        loss_value.backward()
        optimizer = optim.SGD(cur_generator.parameters(), lr=0.1)
        optimizer.step()
        print("output after backward:", output)
        print("input after backward:", input)

def train_nll_loss():
    for i in range(training_steps):
        cur_generator = generator
        #cur_generator = masked_generator
        output=cur_generator(input)
        #output=cur_generator(input, mask0)
        #output=cur_generator(input, mask1)
        #output=cur_generator(input, mask2)
        #output=cur_generator(input, mask3)
        print("target.shape:", target.shape)
        print("output.shape:", output.shape)
        print("input.shape:", input.shape)
        print("target:", target)
        print("input:", input)
        print("output:", output)

        #loss_value = loss_normal(output,target)
        output_s = F.softmax(output, dim=1)
        output_l = torch.log(output_s)
        print("output_s:", output_s)
        print("output_l:", output_l)
        #output_m = output_l * mask1.float()
        #output_m = output_l * mask2.float()
        #output_m = output_l * mask3.float()
        #output_m = output_l * mask4.float()
        output_m = output_l * mask5.float()
        loss_value = F.nll_loss(output_m,target)
        print("output_m:", output_m)
        print("loss_normal(output,target):", loss_value)

        loss_value.backward()
        optimizer = optim.SGD(cur_generator.parameters(), lr=0.1)
        optimizer.step()
        print("output after backward:", output)
        print("input after backward:", input)



def main():
    # train_cross_entropy()
    train_nll_loss()

main()
