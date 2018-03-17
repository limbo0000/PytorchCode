import torch
import torch.nn as nn

# This module acts as an L1 latent state regularizer, adding the
# [gradOutput] to the gradient of the L1 loss. The [input] is copied to
# the [output].

from torch.autograd import Function
class L1Penalty(Function):

    def __init__(self, l1weight, sizeAverage=False, provideOutput=True):
        super(L1Penalty, self).__init__()
        self.l1weight = l1weight
        self.sizeAverage = sizeAverage
        self.provideOutput = provideOutput

    def forward(self, input):
        m = self.l1weight
        if self.sizeAverage:
            m = m / input.nelement()

        loss = m * input.norm(1)
        self.loss = loss
        self.output = input
        return self.output

    def backward(self, input, gradOutput):
        m = self.l1weight
        if self.sizeAverage:
            m = m / input.nelement()

        self.gradInput.resize_as_(input).copy_(input).sign_().mul_(m)

        if self.provideOutput:
            self.gradInput.add_(gradOutput)

        return self.gradInput
  
