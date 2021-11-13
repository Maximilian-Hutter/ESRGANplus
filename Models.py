import torch.nn as nn
import torch
import math

class ResidualDenseResidualBlock(nn.Module):    # Residual Element in DenseBlock
    def __init__(self, filters, res_scale=0.2):
        super(ResidualDenseResidualBlock, self).__init__()

        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]  # Denseblock part
            if non_linearity:
                layers.append(nn.LeakyReLU())
            return nn.Sequential(*layers)


        self.block1 = block(in_features=1 * filters)    # Divided to add a Residual in between the Denseblock
        self.block2 = block(in_features=2 * filters)

        self.blocks1 = [self.block1, self.block2]

        self.block3 = block(in_features=3 * filters) 
        self.block4 = block(in_features=4 * filters)

        self.blocks2 = [self.block3, self.block4]   # blocks1 + blocks2 = full denseblock

        self.Conv = block(in_features=5 * filters, non_linearity=False) # Output Convolution

    def forward(self,x):
        inputs = x
        for block in self.blocks1:
            out=block(inputs)
            inputs = torch.cat([inputs, out],1)

        inputs2 = out.mul(self.res_scale) + x   # add residual to blocks1 out
        residual = inputs2                      # create another residual that is the blocks1 out + x

        for block in self.blocks2:
            out=block(inputs2)
            inputs2 = torch.cat([inputs2, out],1)

        out = self.Conv(out.mul(self.res_scale) + residual) # output convolution

        return out

class ResidualInResidualDenseResidualBlock(nn.Module):  # Residual around the DenseResidualBlocks
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseResidualBlock, self).__init__()

        self.res_scale = res_scale

        self.dense_blocks = nn.Sequential(ResidualDenseResidualBlock(filters), ResidualDenseResidualBlock(filters), ResidualDenseResidualBlock(filters))
        
    def forward(self,x):
        return self.dense_blocks(x).mul(self.res_scale) + x
        
class UpSample(nn.Module):  # Upsampler
    def __init__(self, num_upsample, n_feat):
        super(UpSample, self).__init__()

        modules_body = []
        for _ in range(num_upsample):
            modules_body += [nn.Conv2d(n_feat, 4*n_feat, 3, 1, 1, bias=None)]
            modules_body += [torch.nn.LeakyReLU()]
            modules_body += [torch.nn.PixelShuffle(2)]      

        self.up = torch.nn.Sequential(*modules_body)

    def forward(self,x):
        out = self.up(x)
        return out

class Discriminator(nn.Module): # Discriminator (not part of the Generator)
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
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
        return self.models(img)