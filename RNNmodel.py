import math
import torch

from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np



class GRUcell(Module):
      def __init__(self, bias=False):
          super(GRUcell, self).__init__()
          self.bias = bias
          self.Wxz = nn.Conv2d(18,18,3,1,1,bias=self.bias)
          self.Whz = nn.Conv2d(18,18,3,1,1,bias=self.bias)
          self.Wxr = nn.Conv2d(18,18,3,1,1,bias=self.bias)
          self.Whr = nn.Conv2d(18,18,3,1,1,bias=self.bias)
          self.Wxh = nn.Conv2d(18,18,3,1,1,bias=self.bias)
          self.Whh = nn.Conv2d(18,18,3,1,1,bias=self.bias)
      def reset_parameters(self):
          for weight in self.parameters():
              weight.data.normal_(0.0, 0.02)
      def forward(self, input, h):
          zt = F.sigmoid(self.Wxz(input) + self.Whz(h))
          rt = F.sigmoid(self.Wxr(input) + self.Whr(h))
          h_hat = F.tanh(self.Wxh(input) + rt * (self.Whh(h)))
          output = (1 - zt) * h_hat + zt * h
          return output
class GRU(Module):
      def __init__(self):
          super(GRU, self).__init__()
          self.cell = GRUcell()

      def forward(self, inputs, initial_state):
          time_steps = inputs.size(2)
          state = initial_state
          for t in range(time_steps):
              state = self.cell(inputs[:,:,t,:,:], state)
              if t == 0:
                 output0 = state.view(state.size()[0],18,1,64,64)
                 output = output0
              else:
                 outputt = state.view(state.size()[0],18,1,64,64)
                 output = torch.cat((output,outputt),2)
          return output

