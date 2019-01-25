import torch
import torch.nn as nn
from torch.autograd import Variable

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.mean_square_loss = torch.nn.MSELoss()
        self.fc1 = nn.Linear(4,4)

    def forward(self, x, y):
        y_pred = self.fc1(x)
        y.detach_()
        loss = self.mean_square_loss(y_pred, y)
        return loss

class TestArgMax(nn.Module):
    def __init__(self):
        super(TestArgMax, self).__init__()
        self.mean_square_loss = torch.nn.MSELoss()
        self.fc1 = nn.Linear(4,4)

    def forward1(self, key, value, y):
        key_num, key_index = key.sort()
        print("key_index:", key_index, "key_num:", key_num)
        z = value[key_index]
        y_pred = self.fc1(z)
        y.detach_()
        loss = self.mean_square_loss(y_pred, y)
        return loss

    def forward2(self, key, value, y):
        y_pred = self.fc1(value)
        y.detach_()
        loss = self.mean_square_loss(y_pred, y)
        return loss

    def forward3(self, key, value, y):
        key_num, key_index = key.sort()
        newvalue = self.fc1(value)
        y_pred = newvalue[key_index]
        y.detach_()
        loss = self.mean_square_loss(y_pred, y)
        return loss

    def forward4(self, key, value, y):
        key_num, key_index = key.sort()
        newvalue = self.fc1(value)
        y_pred = newvalue[key_index[-1]]
        y.detach_()
        loss = self.mean_square_loss(y_pred, y[-1])
        return loss

    def forward5(self, key, value, y):
        key_num, key_index = key.sort()
        newvalue = self.fc1(value)
        newvalue.detach()
        y_pred = newvalue[key_index[-1]]
        y.detach_()
        loss = self.mean_square_loss(y_pred, y[-1])
        return loss

    def forward(self, key, value, y):
        #return self.forward1(key, value, y)
        #return self.forward2(key, value, y)
        #return self.forward3(key, value, y)
        #return self.forward4(key, value, y)
        return self.forward5(key, value, y)

def run_test():
    test = Test()

    a=Variable(torch.rand(4,5))
    b=Variable(torch.rand(4,5))
    c=torch.rand(4)
    d=torch.rand(4) # ground-truth

    #cv=Variable(c)
    loss = test(c,d)
    loss.backward()

def run_test_argmax():
    test_argmax = TestArgMax()
    k=torch.rand(4)
    v=torch.rand(4)
    y=torch.rand(4)
    loss = test_argmax(k,v,y)
    loss.backward()

run_test()
run_test_argmax()
