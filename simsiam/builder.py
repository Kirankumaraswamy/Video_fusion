# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs

        self.encoder = base_encoder

        # build a 3-layer projector
        prev_dim = 512
        self.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True)) # output layer

        self.encoder = nn.Sequential(
            self.encoder,
            self.fc
        )

        # self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2, x3):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC
        z3 = self.encoder(x3)  # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC
        p3 = self.predictor(z3)  # NxC

        return p1, p2, p3, z1.detach(), z2.detach(), z3.detach()


class Fusion(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, tuple_len=3):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(Fusion, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs

        self.encoder = base_encoder

        # build a 3-layer projector
        prev_dim = 512
        self.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True)) # output layer



        # self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        # network for clip order
        self.feature_size = pred_dim
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)

        self.fc7 = nn.Linear(self.feature_size * 2, 512)
        pair_num = int(tuple_len * (tuple_len - 1) / 2)
        self.fc8 = nn.Linear(512 * pair_num, self.class_num)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2, x3):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        a1 = self.encoder(x1) # NxC
        z1 = self.fc(a1)
        a2 = self.encoder(x2) # NxC
        z2 = self.fc(a2)
        a3 = self.encoder(x3)  # NxC
        z3 = self.fc(a3)

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC
        p3 = self.predictor(z3)  # NxC


        f = []
        f.append(a1)
        f.append(a2)
        f.append(a2)

        pf = []  # pairwise concat
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                pf.append(torch.cat([f[i], f[j]], dim=1))

        pf = [self.fc7(i) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc8(h)  # logits

        return p1, p2, p3, z1.detach(), z2.detach(), z3.detach(), h
