import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse


class conv1d(nn.Module):
    """
    Includes:
        Convolution, ReLU, Average Pooling
    """

    def __init__(self, n_inputs=1, n_outputs=32, kernel_size=2, stride=1, padding=1):
        super(conv1d, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size=kernel_size, stride=stride, padding=padding)
        self.r1 = nn.ReLU(inplace=True)
        self.p1 = nn.AvgPool1d(kernel_size=3, stride=1)
        self.net = nn.Sequential(self.conv1, self.r1, self.p1)

    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):
    """
    Includes:
        Four convolution layer
    """

    def __init__(self, n_inputs=1, n_outputs=32, kernel_size=2, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        self.conv1 = conv1d(n_inputs, n_outputs, kernel_size, stride, padding)
        self.conv2 = conv1d(n_outputs, n_outputs, kernel_size, stride, padding)
        self.conv3 = conv1d(n_outputs, n_outputs, kernel_size, stride, padding)
        self.conv4 = conv1d(n_outputs, n_outputs, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


class NASNET(nn.Module):
    """
    NASNET
    """

    def __init__(self, in_channels, num_classes, verbose=False):
        super(NASNET, self).__init__()
        self.verbose = verbose
        self.in_channels = in_channels
        self.conv1 = ConvBlock(in_channels, 64, 3, padding=1)
        self.conv2 = ConvBlock(in_channels, 64, 5, padding=2)
        self.conv3 = ConvBlock(self.in_channels, 64, 7, padding=3)
        self.conv4 = ConvBlock(self.in_channels, 64, 9, padding=4)
        self.conv5 = ConvBlock(self.in_channels, 64, 11, padding=5)

        self.dense1 = nn.Linear(16640, 3)
        self.r1 = nn.ReLU(inplace=True)
        self.d1 = nn.Dropout(0.6)
        self.dense2 = nn.Linear(512, 256)
        self.r2 = nn.ReLU(inplace=True)
        self.d2 = nn.Dropout(0.6)
        self.dense3 = nn.Linear(256, 128)
        self.r3 = nn.ReLU(inplace=True)
        self.d3 = nn.Dropout(0.6)

        self.dense4 = nn.Linear(128, 3)
        #self.a4 = nn.Softmax()

        # self.process = nn.Sequential(
        #     self.dense1, self.r1, self.d1, self.dense4)
        # self.process = nn.Sequential(self.dense1, self.r1, self.d1, self.dense2,
        #                              self.r2, self.d2, self.dense3, self.r3, self.d3, self.dense4)
        self.classifier = self.dense1

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out5 = self.conv5(x)

        if self.verbose:
            print('Conv1 output: {}'.format(out1.shape))
            print('Conv2 output: {}'.format(out2.shape))
            print('Conv3 output: {}'.format(out3.shape))
            print('Conv4 output: {}'.format(out4.shape))
            print('Conv5 output: {}'.format(out5.shape))

        out = torch.cat([out1, out2, out3, out4, out5], 1)

        if self.verbose:
            print('Concatenate output: {}'.format(out.shape))

        out = out.view(out.shape[0], -1)

        if self.verbose:
            print('Flattern output: {}'.format(out.shape))

        return self.classifier(out)
