from _typeshed import OpenTextModeUpdating
import torch.nn as nn
import torch
import math

class ResidualDenseResidualBlock(nn.Module):
    def __init(self,num_DenseBlock, filters, res_scale=0.2, in_features=1):
        super(ResidualDenseResidualBlock, self).__init__()

        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers.append(nn.LeakyReLU())
            return nn.Sequential(*layers)

        in_features = in_features + num_DenseBlock

        self.block1 = block(in_features=1 * filters)
        self.block2 = block(in_features=2 * filters)

        self.blocks1 = [self.block1, self.block2]

        self.block3 = block(in_features=3 * filters)
        self.block4 = block(in_features=4 * filters)

        self.blocks2 = [self.block3, self.block4]

        self.Conv = block(in_features=5 * filters, non_linearity=False)

    def forward(self,x):
        inputs = x
        for block in self.blocks1:
            out=block(inputs)
            inputs = torch.cat([inputs, out],1)

        inputs2 = out.mul(self.res_scale) + x
        residual = inputs2

        for block in self.blocks2:
            out=block(inputs2)
            inputs2 = torch.cat([inputs2, out],1)

        out = self.Conv(out.mul(self.res_scale) + residual)

        return out

class ResidualInResidualDenseResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseResidualBlock, self)

        self.res_scale = res_scale

        self.dense_blocks = nn.Sequential(ResidualDenseResidualBlock(filters), ResidualDenseResidualBlock(filters), ResidualDenseResidualBlock(filters))
        
    def forward(self,x):
        return self.dense_blocks(x).mul(self.res_scale) + x
        
class UpSample(nn.Module):
    def __init__(self, num_upsample, n_feat):
        super(UpSample).__init__()

        modules = []
        for _ in range(num_upsample):
            modules.append(nn.Conv2d(n_feat, 4*n_feat, 3, 1, 1, bias=None, activation=None, norm=None))
            modules.append(torch.nn.LeakyReLU())
            modules.append(torch.nn.PixelShuffle(2))            

        self.up = torch.nn.Sequential(*modules)

    def forward(self,x):
        out = self.up(x)
        return out

class ResnetBlock(nn.Module):   # resnet block might not be like this
    def __init__(self, filters, res_scale=0.2):
        super(ResnetBlock).__init__()

        self.res_scale = res_scale
        

    def forward(self,x):
        residual = x
        x = self.ConvB1(x)
        x = self.ConvB2(x)
        x = torch.add(residual,x)
        residual = x
        x = self.ConvB3(x)
        x = self.ConvB4(x)
        out = torch.add(residual, x)

        return out

class Discriminator(nn.Module):
    def __init(self, input_shape):
        super().__init_()

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