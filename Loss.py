import torch
import torch.nn as nn
from torchvision.model.vgg import vgg19
import math

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, predicted_high_resolution, high_resolution):
        perception_loss = self.l1Loss(self.loss_network(high_resolution), self.loss_network(predicted_high_resolution))

        return perception_loss