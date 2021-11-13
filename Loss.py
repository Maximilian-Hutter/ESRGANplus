import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19
import math

class FeatureExtractor(nn.Module):    # Loss to optimize for Human perception Visual quality
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg = vgg19(pretrained=True)    # pretrained vgg 19
        self.vgg19_54 = nn.Sequential(*list(vgg.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)