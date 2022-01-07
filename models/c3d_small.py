"""C3D"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class C3DSMALL(nn.Module):
    """C3D with BN and pool5 to be AdaptiveAvgPool3d(1)."""

    def __init__(self, with_classifier=False, return_conv=False, num_classes=101):
        super(C3DSMALL, self).__init__()
        self.with_classifier = with_classifier
        self.num_classes = num_classes
        self.return_conv = return_conv

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3a = nn.BatchNorm3d(256)
        self.relu3a = nn.ReLU()
        self.conv3b = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3b = nn.BatchNorm3d(512)
        self.relu3b = nn.ReLU()

        if self.return_conv:
            self.feature_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 9216
            # self.feature_pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) 4182

        self.pool5 = nn.AdaptiveAvgPool3d(1)

        if self.with_classifier:
            self.linear = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.relu3a(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = self.relu3b(x)

        if self.return_conv:
            x = self.feature_pool(x)
            # print(x.shape)
            return x.view(x.shape[0], -1)

        x = self.pool5(x)
        x = x.view(-1, 512)

        if self.with_classifier:
            x = self.linear(x)

        return x


if __name__ == '__main__':
    c3d_small = C3DSMALL()