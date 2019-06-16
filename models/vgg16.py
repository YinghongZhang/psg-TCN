import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """
    VGG16 network

    """

    def __init__(self, features, num_classes, verbose=False):
        super(VGG, self).__init__()

        self.features = features
        self.verbose = verbose

        self.classifier = nn.Sequential(
            # nn.Linear(512, 3),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(512, 512),
            # nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        # 512 * 7* 7, 4096, 4096, 4096, 3
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        if self.verbose:
            print("Conv layers: {}".format(x.shape))
        # 第一项为batch size，其余应该叠加在一起，所以用-1表示
        x = x.view(x.size(0), -1)
        if self.verbose:
            print("Flattern: {}".format(x.shape))
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            # 不同的网络类型初始化均不同，比如Conv2d卷积层的权值初始化为均值为0，偏置为0
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model
