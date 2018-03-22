#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn.init as inital
def weights_init_normal(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Linear):
        inital.normal(m.weight.data,0,0.02)
        inital.normal(m.bias.data,0,0.02)
def weights_init_xavier(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Linear):
        inital.xavier_normal(m.weight.data)
        inital.xavier_normal(m.bias.data)

def weights_init_kaiming(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Linear):
        inital.kaiming_normal(m.weight.data)
        inital.kaiming_normal(m.bias.data)

