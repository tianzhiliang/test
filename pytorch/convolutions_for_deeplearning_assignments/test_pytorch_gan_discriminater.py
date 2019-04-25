import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        #
        # 3*32*32 -> 32*16*16
        # input_size: 32*32     output_size: 16*16
        # in_channels: 3        out_channels:128
        self.conv1 = nn.Conv2d(3, 32, (17, 17), stride=1)
        # 32*16*16 -> 64*8*8
        self.conv2 = nn.Conv2d(32,64,(9,9),stride=1)
        # 64*8*8 -> 128*4*4
        self.conv3 = nn.Conv2d(64,128,(5,5),stride=1)
        # 128*4*4 -> 1*1*1
        self.conv4 = nn.Conv2d(128,1,(4,4),stride=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        print("out: ", out.size())
        out = F.relu(self.conv2(out))
        print("out: ", out.size())
        out = F.relu(self.conv3(out))
        print("out: ", out.size())

        out = self.conv4(out).squeeze()
        print("out: ", out.size())
        out = F.sigmoid(out)
        print("out: ", out.size())

x = torch.randn(128,3,32,32)
print("input:", x.size())
model = Model()
model.forward(x)
