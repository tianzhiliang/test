import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, \
        #        padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        # output_size = (input_size - 1) * stride + output_padding - 2 * padding + kernel_size
        #
        # 100*1*1 -> 128*4*4
        # in_channels: 100   out_channels: 128 
        # input_size: 1*1    output_size: 4*4
        self.deconv1 = nn.ConvTranspose2d(100, 128, (4, 4), stride=1)

        # 128*4*4 -> 64*8*8
        self.deconv2 = nn.ConvTranspose2d(128, 64, (5, 5), stride=1)

        # 64*8*8 -> 32*16*16
        self.deconv3 = nn.ConvTranspose2d(64, 32, (9, 9), stride=1)

        # 32*16*16 -> 3*32*32
        self.deconv4 = nn.ConvTranspose2d(32, 3, (17, 17), stride=1)

    def forward(self, z):
        out = F.relu(self.deconv1(z))
        print("out: ", out.size())
        out = F.relu(self.deconv2(out))
        print("out: ", out.size())
        out = F.relu(self.deconv3(out))
        print("out: ", out.size())
        out = F.tanh(self.deconv4(out))
        print("out: ", out.size())

x = torch.randn(16,100,1,1)
print("input:", x.size())
model = Model()
model.forward(x)
