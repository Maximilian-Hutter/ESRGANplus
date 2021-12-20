import torch 
import torch.nn as nn
from noise_generator import *
from Models import ResidualInResidualDenseResidualBlock, UpSample

class ESRGANplus(nn.Module):    # generator
    def __init__(self, channels, filters, hr_shape, n_resblock, num_upsample = 4,  res_scale=0.2):
        super(ESRGANplus, self).__init__()
        
        self.n_resblock = n_resblock
        self.Conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)

        layers= [ResidualInResidualDenseResidualBlock(filters),GaussianNoiseGenerator(hr_shape)]

        self.RRDRB = nn.Sequential(*layers)

        self.Conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        self.Upsample = UpSample(num_upsample, filters)
        
        self.Conv3 = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=3, stride=1,padding=1), nn.LeakyReLU(),nn.Conv2d(filters, channels, kernel_size=3, stride=1,padding=1))
        

    def forward(self, input):
        out1 = self.Conv1(input)

        for _ in range(self.n_resblock):
            out1 = self.RRDRB(out1)
            #print("RRDRB Num: {}".format(_))
            out2 = out1

        out2 = self.Conv2(out2)        
        out = torch.add(out1, out2)
        out = self.Upsample(out)
        #print(out.size())
        out = self.Conv3(out)

        return out