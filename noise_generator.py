import torch
from torch.autograd.variable import Variable
import torch.nn as nn

# sigma = relatie standard deviation
# is_realtiv_detach = whether to detach the variable before compute

class GaussianNoiseGenerator(nn.Module):    # Gaussian Noise generator
    def __init__(self):
        super(GaussianNoiseGenerator, self).__init__()



    def forward(self,input):

        out = input + (0.1**0.5)*torch.randn(512, device=torch.device('cuda', 0))

        return out