import torch
from torch.autograd.variable import Variable
import torch.nn as nn

# sigma = relatie standard deviation
# is_realtiv_detach = whether to detach the variable before compute

class GaussianNoiseGenerator(nn.Module):    # Gaussian Noise generator
    def __init__(self, hr_shape, upsample):
        super(GaussianNoiseGenerator, self).__init__()

        self.hr_height, self.hr_width, = hr_shape
        self.upsample = upsample

    def forward(self,input):
        
        if torch.cuda.is_available():
            out = input + (0.1**0.5)*torch.randn((self.hr_height, self.hr_width), device=torch.device('cuda', 0))

        else:
            out = input + (0.1**0.5)*torch.randn((self.hr_height, self.hr_width), device = torch.device('cpu',0))
        return out