#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
class RankLoss(nn.Module):
    def __init__(self, dis=0.5):
        super(RankLoss,self).__init__()
        self.dis = dis
    def forward(self, normal, reverse, static):
        nr = 1 - F.cosine_similarity(normal, reverse)
        ns = 1 - F.cosine_similarity(normal, static)
        loss = F.relu(nr - ns + self.dis)
        return torch.mean(loss)


