import numpy as np
import torch
import torch.nn as nn

import torch
from torch.autograd import Variable


def kernel(kernel, n_channels):
    return np.tile(kernel, (n_channels, 1, 1)).swapaxes(0,1).swapaxes(1,2)
def initKernel():

    n = 3
    gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    kerx = np.stack([gx,gx,gx])
    kery = np.stack([gy,gy,gy])
    ker = np.stack([kerx, kery])
    return torch.FloatTensor(ker)
class Sobel(nn.Module):
      def __init__(self):
          super(Sobel, self).__init__()
          self.sobel = nn.Conv2d(3,2,(3,3),bias=False)
          self.sobel.weight.requires_grad = False
          self.data = initKernel()
          self.sobel.weight.data = self.data.cuda()
      def forward(self, input):
          #self.sobel.weight.data = self.data
          timesteps = input.size()[2]
          for time in range(timesteps):
              if time < 1:
                 output0 = self.sobel(input[:,:,time,:,:])
                 size = output0.size()
                 output = output0.view(size[0], size[1], 1, size[2], size[3])
                 continue
              outputi = self.sobel(input[:,:,time,:,:])
              size = outputi.size()
              outputi = outputi.view(size[0], size[1], 1, size[2], size[3])
              output = torch.cat((output, outputi), 2)
          return output
