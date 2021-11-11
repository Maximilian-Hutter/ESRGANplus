from numpy.random.mtrand import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import glob
import random
import os
import numpy as np

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)
class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width, = hr_shape

        self.lr_transforms = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transforms = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transforms(img)
        img_hr = self.hr_transforms(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)