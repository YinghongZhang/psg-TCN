import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import math


def conv_relu(in_channels, out_channles, kernel, stride=1, padding=0):
    """
    Includes:
        Convolution, Batch normalization, ReLU
    """
    layer = nn.Sequential(
        nn.Conv1d(in_channels, out_channles, kernel, stride, padding),
        nn.BatchNorm1d(out_channles, eps=1e-3),
        nn.ReLU(True)
    )
    return layer


class Inception(nn.Module):
    """
    Inception layer
    """

    def __init__(self, in_channels, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(Inception, self).__init__()

        self.conv1 = conv_relu(in_channels, out1_1, 1)

        self.conv2 = nn.Sequential(
            conv_relu(in_channels, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, 1, 1),
        )

        self.conv3 = nn.Sequential(
            conv_relu(in_channels, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, 1, 2),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            conv_relu(in_channels, out4_1, 1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.branch_pool(x)

        out = torch.cat((out1, out2, out3, out4), 1)

        return out


class GoogleNet(nn.Module):
    """
    One dimension Google Net
    """

    def __init__(self, in_channels=1, num_classes=3, verbose=False):
        super(GoogleNet, self).__init__()
        self.verbose = verbose
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            conv_relu(in_channels, out_channles=64,
                      kernel=7, stride=1, padding=2),
            nn.MaxPool1d(3, 2)
        )

        self.block2 = nn.Sequential(
            conv_relu(64, 128, kernel=1),
            conv_relu(128, 256, kernel=3, padding=1),
            nn.MaxPool1d(3, 2),
        )

        self.block3 = nn.Sequential(
            Inception(256, 128, 96, 128, 16, 64, 64),
            Inception(384, 128, 128, 192, 32, 192, 128),
            nn.MaxPool1d(3, 2)
        )

        self.block4 = nn.Sequential(
            Inception(640, 192, 96, 208, 16, 96, 128),
            Inception(624, 160, 112, 224, 24, 128, 128),
            Inception(640, 128, 128, 256, 24, 128, 128),
            Inception(640, 112, 144, 288, 32, 128, 128),
            Inception(656, 256, 160, 320, 32, 160, 160),
            nn.MaxPool1d(3, 2)
        )

        self.classifier = nn.Linear(1792, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            # 不同的网络类型初始化均不同，比如Conv2d卷积层的权值初始化为均值为0，偏置为0
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.xavier_uniform(m.weight)
                # nn.init.kaiming_normal(m.weight)
                # nn.init.kaiming_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                # nn.init.xavier_uniform(m.weight)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform(m.weight)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block4 output: {}'.format(x.shape))
        # x = self.block5(x)
        # if self.verbose:
        #     print('block5 output: {}'.format(x.shape))

        x = x.view(x.shape[0], -1)
        if self.verbose:
            print("Flattern output: {}".format(x.shape))

        x = self.classifier(x)
        return x
