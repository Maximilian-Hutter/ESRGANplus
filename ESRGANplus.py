import torch 
import torch.nn as nn
from noise_generator import *
from Models import ResidualInResidualDenseResidualBlock, UpSample

class ESRGANplus(nn.Module):    # generator
    def __init__(self, channels, filters, num_upsample = 2, n_resblock = 16, res_scale=0.2):
        super(ESRGANplus, self).__init__()
        
        self.Conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        
        self.RRDRB = nn.Sequential(torch.add(ResidualInResidualDenseResidualBlock(filters),GaussianNoiseGenerator()) for _ in range(n_resblock))

        self.Conv2 = nn.Conv2d(filters, filters, kerne_size=3, stride=1, padding=1)

        self.Upsample = UpSample(num_upsample, filters)

        self.Conv3 = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=3, stride=1,padding=1), nn.LeakyReLU(),nn.Conv2d(filters, channels, kernel_size=3, stride=1,padding=1))
        
    def forward(self, x):
        
        out1 = self.Conv1(x)
        out = self.RRDRB(out1)
        out2 = self.Conv2(out)        
        out = torch.add(out1, out2)
        out = self.Upsample(out)
        out = self.Conv3(out)

        return out