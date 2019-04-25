import os,random,sys,math
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

use_cuda = torch.cuda.is_available()

def download_process_data(batch_size):
    mnist_data_path="./mnist_data/"
    trans = transforms.Compose([transforms.ToTensor()])
    #trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.MNIST(root=mnist_data_path, train=True, transform=trans, download=False)
    test_set = dset.MNIST(root=mnist_data_path, train=False, transform=trans, download=False)
    #train_set = dset.MNIST(root=mnist_data_path, train=True, download=True)
    #test_set = dset.MNIST(root=mnist_data_path, train=False, download=True)
    #train_set = dset.MNIST(root=mnist_data_path, train=True, download=False)
    #test_set = dset.MNIST(root=mnist_data_path, train=False, download=False)

    train_data = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)
    test_data = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)
    return train_data, test_data

def count_right_num(pred, real):
    cnt, right = 0, 0
    #print("pred:", pred, "real:", real)
    #print("size pred:", pred.size(), "real:", real.size())
    for p, r in zip(pred, real):
        print("p:", p, "r:", r)
        if p == r:
            right +=1 
        cnt += 1
    return right, cnt, right/cnt

class LeNet2(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(x.size(0), -1)
        output = self.fc(output)
        return output

class LeNet3(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1*28*28 -> 6*28*28
        self.conv1 = nn.Conv2d(1, 6, (1, 1), stride=1)
        # 6*28*28 -> 6*14*14
        self.pool1 = nn.AvgPool2d((2, 2), stride=2)
        # 6*14*14 -> 16*10*10
        self.conv2 = nn.Conv2d(6,16,(5,5),stride=1)
        # 16*10*10 -> 16*5*5
        self.pool2 = nn.AvgPool2d((2, 2), stride=2)
        # 16*5*5 -> 120*1*1
        self.conv3 = nn.Conv2d(16,120,(5,5),stride=1) 
        # 120 -> 84
        self.fc1 = nn.Linear(120, 84)
        # 84 -> 10
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = F.tanh(self.conv3(x))
        x = x.view(x.shape[:2])
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self, inputdim, hiddim, outdim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inputdim, hiddim)
        self.fc2 = nn.Linear(hiddim, hiddim)
        self.fc3 = nn.Linear(hiddim, hiddim)
        self.fc4 = nn.Linear(hiddim, outdim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x2 = self.relu(self.fc2(x1))
        x3 = self.relu(self.fc3(x2))
        x4 = self.fc4(x3)
        return x4

class CrossEntropy():
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred_y, real_y):
        return self.loss(pred_y, real_y)

def knn(train_data, test_data, n_neighbors):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric='manhattan')
    train_x = [td[0].data.numpy()[0][0].ravel() for td in train_data]
    train_y = [td[1].data.numpy()[0] for td in train_data]
    #for i in train_x:
    #    print("train_x:", i)
    #print("train_y:", train_y)
    neigh.fit(train_x, train_y)

    acc, cnt = 0, 0
    #test_x = [td[0].data.numpy()[0][0].ravel() for td in test_data]
    #test_y = [td[1].data.numpy()[0] for td in test_data]
    test_x = [td[0].data.numpy()[0][0].ravel() for td in train_data]
    test_y = [td[1].data.numpy()[0] for td in train_data]
    pred_y = neigh.predict(test_x)
    for pred, test in zip(pred_y, test_y):
        print("pred:", pred, "test:", test)
        if pred == test:
            acc += 1
        cnt += 1

    print("K:", n_neighbors, "right:", acc, "total:", cnt, "acc:", acc/cnt)

def run_knn():
    train_data, test_data = download_process_data(batch_size=1)
    #for n_neighbors in range(1, 11):
    #    knn(train_data, test_data, n_neighbors)
    knn(train_data, test_data, 1)

def mlp(train_data, test_data, inputdim, hiddim, outdim):
    model = MLP(inputdim, hiddim, outdim)
    if use_cuda:
        model = model.cuda()
        print("use use_cuda")
    cross_entropy_loss = CrossEntropy()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    epoch = 5
    for e in range(0, epoch):
        for i, (train_x, train_y) in enumerate(train_data):
            if use_cuda:
                train_x, train_y = train_x.cuda(), train_y.cuda()
            #train_x = train_x[0][0].flatten()
            train_x = train_x.flatten(1)
            #train_y = train_y[0]
            train_x, train_y = Variable(train_x), Variable(train_y)
            optimizer.zero_grad()
            pred_out = model.forward(train_x)
            #print("pred_out:", pred_out, "train_y:", train_y)
            #print("size pred_out:", pred_out.size(), "train_y:", train_y.size())
            loss = cross_entropy_loss.forward(pred_out, train_y)
            print("hiddim:", hiddim, "epoch:", e, "batch:", i, "loss:", loss.item())
            loss.backward()
            optimizer.step()

    tright, tcnt = 0, 0
    for i, (test_x, test_y) in enumerate(test_data):
        if use_cuda:
            test_x, test_y = test_x.cuda(), test_y.cuda()
        test_x = test_x.flatten(1)
        test_y = test_y
        pred_out = model.forward(test_x)
        pred_y_prob, pred_y = torch.max(pred_out, dim=1)
        right, cnt, acc = count_right_num(pred_y, test_y)
        print("hiddim:", hiddim, "right:", right, "cnt:", cnt, "acc:", acc)
        tright += right
        tcnt += cnt
    print("total hiddim:", hiddim, "right:", tright, "tcnt:", tcnt, "acc:", tright/tcnt)

def run_mlp():
    #train_data, test_data = download_process_data(batch_size=1)
    train_data, test_data = download_process_data(batch_size=128)
    hiddims = [4, 8, 16, 32, 64, 128, 256]
    for hiddim in hiddims:
        mlp(train_data, test_data, 784, hiddim, 10)

def lenet(train_data, test_data):
    model = LeNet()
    if use_cuda:
        model = model.cuda()
        print("use use_cuda")
    cross_entropy_loss = CrossEntropy()
    #optimizer = optim.Adam(model.parameters(), lr=2e-3)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    epoch = 5
    for e in range(0, epoch):
        for i, (train_x, train_y) in enumerate(train_data):
            if use_cuda:
                train_x, train_y = train_x.cuda(), train_y.cuda()
            #train_x = train_x.flatten(1)
            #print("train_x:", train_x)
            #print("size train_x:", train_x.size())
            #print("train_y:", train_y)
            #print("size train_y:", train_y.size())
            train_x, train_y = Variable(train_x), Variable(train_y)
            optimizer.zero_grad()
            pred_out = model.forward(train_x)
            loss = cross_entropy_loss.forward(pred_out, train_y)
            print("epoch:", e, "batch:", i, "loss:", loss.item())
            loss.backward()
            optimizer.step()

    tright, tcnt = 0, 0
    for i, (test_x, test_y) in enumerate(test_data):
        if use_cuda:
            test_x, test_y = test_x.cuda(), test_y.cuda()
        #test_x = test_x.flatten(1)
        pred_out = model.forward(test_x)
        pred_y_prob, pred_y = torch.max(pred_out, dim=1)
        right, cnt, acc = count_right_num(pred_y, test_y)
        print("right:", right, "cnt:", cnt, "acc:", acc)
        tright += right
        tcnt += cnt
    print("total right:", tright, "tcnt:", tcnt, "acc:", tright/tcnt)

def run_lenet():
    train_data, test_data = download_process_data(batch_size=128)
    lenet(train_data, test_data)

def main():
    #run_knn()
    #run_mlp()
    run_lenet()

main()
