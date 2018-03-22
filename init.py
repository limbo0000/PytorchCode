#!/usr/bin/env python
# coding=utf-8
import torch.nn as nn
import nn.init.xavier_normal as xavier
import nn.init.kaiming_normal as kaiming
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        xavier(m.weight.data)
        xavier(m.bias.data)

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        kaiming(m.weight.data)
        kaiming(m.bias.data)

