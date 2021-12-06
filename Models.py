import torch.nn as nn
import torch
import math

class ResidualDenseResidualBlock(nn.Module):    # Residual Element in DenseBlock
    def __init__(self, filters, res_scale=0.2, NetType = 'ESRGAN'):
        super(ResidualDenseResidualBlock, self).__init__()

        self.res_scale = res_scale
        self.NetType = NetType

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]  # Denseblock part
            if non_linearity:
                layers.append(nn.LeakyReLU())
            return nn.Sequential(*layers)

        self.block1 = block(in_features=1 * filters)    # Divided to add a Residual in between the Denseblock
        self.block2 = block(in_features=2 * filters)
        self.block3 = block(in_features=3 * filters) 
        self.block4 = block(in_features=4 * filters)
        self.block5 = block(in_features=5 * filters, non_linearity=False) # Output Convolution
        self.blocks = [self.block1, self.block2, self.block3, self.block4,self.block5]

    def forward(self,x):
        inputs = x

        if(self.NetType == 'ESRGANplus'):
            for block in self.blocks:   # if memory problems split into multiple for loops or no loop at all
                if block == 2:
                    inputs += x
                    residual = inputs
                if block == 4:
                    inputs += residual
                out=block(inputs)
                inputs = torch.cat([inputs, out],1)
        if(self.NetType == 'ESRGAN'):
            for block in self.blocks:
                out=block(inputs)
                inputs = torch.cat([inputs,out],1)

        return out.mul(self.res_scale) + x

class ResidualInResidualDenseResidualBlock(nn.Module):  # Residual around the DenseResidualBlocks
    def __init__(self, filters, res_scale=0.2,num_DenseBlocks = 3):
        super(ResidualInResidualDenseResidualBlock, self).__init__()

        self.res_scale = res_scale
        self.num_DenseBlocks = num_DenseBlocks

        self.dense_block = ResidualDenseResidualBlock(filters)

    def forward(self,x):
        for _ in range(self.num_DenseBlocks):
            x = self.dense_block(x)
            out = x
        return out.mul(self.res_scale) + x
        
class UpSample(nn.Module):  # Upsampler
    def __init__(self, num_upsample, filters):
        super(UpSample, self).__init__()

        self.num_upsample = num_upsample
        modules_body = []
        modules_body += [nn.Conv2d(filters, 4*filters, 3, 1, 1, bias=None)]
        modules_body += [torch.nn.LeakyReLU()]
        modules_body += [torch.nn.PixelShuffle(2)]      

        self.up = torch.nn.Sequential(*modules_body)

    def forward(self,x):
        print(x.size())
        for _ in range(self.num_upsample):
            x = self.up(x)
            out = x

            print(out.size())
        return out

class Discriminator(nn.Module): # Discriminator (not part of the Generator)
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape # might create the content loss img width error
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2**4)
        self.output_shape = (1,patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i==0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)