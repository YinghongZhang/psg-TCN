from torch import nn
import torch as t
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    # 实现子module: Residual    Block
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    # 实现主module:ResNet34
    # ResNet34包含多个layer,每个layer又包含多个residual block
    # 用子module实现residual block , 用 _make_layer 函数实现layer
    def __init__(self, num_classes=3, verbose=False):
        super(ResNet, self).__init__()
        self.verbose = verbose

        self.pre = nn.Sequential(
            nn.Conv1d(1, 64, 7, 2, 3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1)
        )
        # 重复的layer,分别有3,4,6,3个residual block
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 构建layer,包含多个residual block
        shortcut = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm1d(outchannel))

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        if self.verbose:
            print("Pre {}".format(x.shape))
        x = self.layer1(x)
        if self.verbose:
            print("Layer1 {}".format(x.shape))
        x = self.layer2(x)
        if self.verbose:
            print("Layer2 {}".format(x.shape))
        x = self.layer3(x)
        if self.verbose:
            print("Layer3 {}".format(x.shape))
        x = self.layer4(x)
        if self.verbose:
            print("Layer4 {}".format(x.shape))

        #x = F.avg_pool2d(x, 7)
        # if self.verbose:
        #     print("AvgPool {}".format(x.shape))
        x = x.view(x.size(0), -1)
        if self.verbose:
            print("Flattern {}".format(x.shape))
        return self.fc(x)
