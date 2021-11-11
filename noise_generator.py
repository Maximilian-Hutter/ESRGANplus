import torch
from torch.autograd.variable import Variable
import torch.nn as nn

# sigma = relatie standard deviation
# is_realtiv_detach = whether to detach the variable before compute

class GaussianNoiseGenerator(nn.module):    # Gaussian Noise generator
    def __init__(self, input = 0, sigma = 0.1, is_relative_detach=True):
        super(GaussianNoiseGenerator, self).__init__()

        self.sigma = sigma
        self.is_relativ_detach = is_relative_detach
        self.register_buffer('noise', torch.tensr(0))

    def forward(self,input):
        if not self.training and self.sigma != 0:
            scale = self.sigma * input.detach() if self.is_relative.detach else self.sigma + input
            self.noise.data.normal_(0,std=self.std)
            sampled_noise = self.noise.expand(*input.size()).float().normal() * scale
            x = input + sampled_noise
        return x