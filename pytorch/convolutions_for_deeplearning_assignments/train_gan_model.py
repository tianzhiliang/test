#  COMP 6211D & ELEC 6910T , Assignment 3
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator       --> Used in the vanilla GAN in Part 1
#   - CycleGenerator    --> Used in the CycleGAN in Part 2
#   - DCDiscriminator   --> Used in both the vanilla GAN and CycleGAN (Parts 1 and 2)
#
# For the assignment, you are asked to create the architectures of these three networks by
# filling in the __init__ methods in the DCGenerator, CycleGenerator, and DCDiscriminator classes.
# Note that the forward passes of these models are provided for you, so the only part you need to
# fill in is __init__.

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

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):
    def __init__(self, noise_size = 100, conv_dim = 128):
        super(DCGenerator, self).__init__()

        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, \
        #        padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        # output_size = (input_size - 1) * stride + output_padding - 2 * padding + kernel_size
        #
        # 100*1*1 -> 128*4*4
        # in_channels: 100   out_channels: 128 
        # input_size: 1*1    output_size: 4*4
        #self.deconv1 = nn.ConvTranspose2d(100, 128, (4, 4), stride=1)
        #self.deconv1 = deconv(100, 128, (4, 4), stride=1, padding=0, batch_norm=True)
        self.deconv1 = deconv(noise_size, 128, (4, 4), stride=1, padding=0, batch_norm=True)

        # 128*4*4 -> 64*8*8
        #self.deconv2 = nn.ConvTranspose2d(128, 64, (5, 5), stride=1)
        self.deconv2 = deconv(128, 64, (5, 5), stride=1, padding=0, batch_norm=True)

        # 64*8*8 -> 32*16*16
        #self.deconv3 = nn.ConvTranspose2d(64, 32, (9, 9), stride=1)
        self.deconv3 = deconv(64, 32, (9, 9), stride=1, padding=0, batch_norm=True)

        # 32*16*16 -> 3*32*32
        #self.deconv4 = nn.ConvTranspose2d(32, 3, (17, 17), stride=1)
        self.deconv4 = deconv(32, 3, (17, 17), stride=1, padding=0, batch_norm=False)


    def forward(self, z):
        """Generates an image given a sample of random noise.

            Input
            -----
                z: BS x noise_size x 1 x 1   -->  16x100x1x1

            Output
            ------
                out: BS x channels x image_width x image_height  -->  16x3x32x32
        """

        out = F.relu(self.deconv1(z))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = F.tanh(self.deconv4(out))
        return out


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self):
        super(CycleGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # 1. Define the encoder part of the generator (that extracts features from the input image)

        # 2. Define the transformation part of the generator
        
        # 3. Define the decoder part of the generator (that builds up the output image from features)


    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = F.relu(self.resnet_block(out))

        out = F.relu(self.deconv1(out))
        out = F.tanh(self.deconv2(out))

        return out


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim = 128):
        super(DCDiscriminator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################
        # 3*32*32 -> 32*16*16
        #self.conv1 = nn.Conv2d(3, 32, (17, 17), stride=1)
        self.conv1 = conv(3, 32, (17, 17), stride=1, padding=0, batch_norm=True)
        # 32*16*16 -> 64*8*8
        #self.conv2 = nn.Conv2d(32,64,(9,9),stride=1)
        self.conv2 = conv(32,64,(9,9),stride=1, padding=0, batch_norm=True)
        # 64*8*8 -> 128*4*4
        #self.conv3 = nn.Conv2d(64,128,(5,5),stride=1)
        self.conv3 = conv(64,128,(5,5),stride=1, padding=0, batch_norm=True)
        # 128*4*4 -> 1*1*1
        #self.conv4 = nn.Conv2d(128,1,(4,4),stride=1)
        self.conv4 = conv(128,1,(4,4),stride=1, padding=0, batch_norm=False)

    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = self.conv4(out).squeeze()
        out = F.sigmoid(out)
        return out
