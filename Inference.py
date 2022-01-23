import torch
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode
from ESRGANplus import *
from Models import *
from torchvision import transforms, utils
from PIL import Image
import argparse
import torchvision


parser = argparse.ArgumentParser(description='PyTorch ESRGANplus')
parser.add_argument('--modelpath', type=str, default='weights/2x_ESRGANplus.pth', help=("path to the model .pth files"))
parser.add_argument('--inferencepath', type=str, default='../../data/inferenceImage/', help=("Path to image folder"))
parser.add_argument('--imagename', type=str, default='001.jpg', help=("filename of the image"))
parser.add_argument('--hr_height', type=int, default= 320, help='high res. image height')
parser.add_argument('--hr_width', type=int, default= 480, help='high res. image width')
parser.add_argument('--gpu_mode', type=bool, default=False, help=('enable cuda'))
parser.add_argument('--channels',type=int, default=3, help='number of channels R,G,B for img / number of input dimensions 3 times 2dConv for img')
parser.add_argument('--filters',type=int, default=64, help='number of filters')
parser.add_argument('--n_resblock', type=int, default=7, help='number of Res Blocks')
parser.add_argument('--upsample', type=int, default=2, help="super resolution upscale factor")

      

opt = parser.parse_args()

PATH = opt.modelpath
imagepath = (opt.inferencepath + opt.imagename)
image = Image.open(imagepath)
imageLR = Image.open(imagepath)
data_transforms = transforms.Compose([transforms.Resize((opt.hr_height, opt.hr_width)), transforms.ToTensor()])
LRtransform = transforms.Compose([transforms.Resize((opt.hr_height * 4, opt.hr_width * 4)), transforms.ToTensor()])

imageLR = LRtransform(imageLR)

if opt.gpu_mode == False:
    image = data_transforms(image).unsqueeze(0)

if opt.gpu_mode:
    image = data_transforms(image).unsqueeze(0).cuda()

utils.save_image(imageLR,'../../results/ESRGANplus/LR_image.jpg')
hr_shape = (opt.hr_height, opt.hr_width)
model=ESRGANplus(opt.channels, filters=opt.filters,hr_shape=hr_shape, n_resblock = opt.n_resblock, upsample = opt.upsample)

if opt.gpu_mode == False:
    device = torch.device('cpu')

if opt.gpu_mode:
    	device = torch.device('cuda:0')

model.load_state_dict(torch.load(PATH,map_location=device))

model.eval()

out = model(image)
utils.save_image(out,'../../results/ESRGANplus/HR_image.png')