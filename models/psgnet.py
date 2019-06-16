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


class PSGNet(nn.Module):
    """
    Self define network
    """

    def __init__(self, in_channels, out_classes=3, verbose=False):
        super(PSGNet, self).__init__()
        self.verbose = verbose
        self.block1 = nn.Sequential(
            conv_relu(in_channels, out_channles=64,
                      kernel=7, stride=1, padding=2),
            nn.MaxPool1d(3, 2)
        )

        self.conv2 = conv_relu(64, 128, kernel=1)

        self.conv3 = conv_relu(128, 256, kernel=3, padding=1)
        self.mp = nn.MaxPool1d(3, 2)

        self.block2 = nn.Sequential(
            conv_relu(64, 128, kernel=1),
            conv_relu(128, 256, kernel=3, padding=1),
            nn.MaxPool1d(3, 2),
        )

        self.classifier = nn.Sequential(nn.Linear(3328, 512),
                                        nn.Linear(512, 256),
                                        nn.Linear(256, 128),
                                        nn.Linear(128, out_classes))

    def forward(self, x):
        x = self.block1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mp(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
