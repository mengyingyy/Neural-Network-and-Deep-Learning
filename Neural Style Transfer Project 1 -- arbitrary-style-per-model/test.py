# This model is used as a controlled group
# usage: python test.py -c [content_image_file_path] -s [style_image_file_path]
# modified from https://github.com/Joanhxp/Style-Swap/blob/master/main.py
# fixed some bugs of model input and model loading

import os
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from model import *

args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-c','--content', default="./content", type=str, help='Content dataset folder path')
args.add_argument('-s','--style', default="./style", type=str, help='Style dataset folder path')
args.add_argument('-o','--output', default="./outputs", type=str, help='Output image folder path')
args.add_argument('-p','--patchsize', default=3, type=int, help='Patch size of style swap')
args.add_argument('-r','--relulevel', default=3, type=int, help='Relu-level of style swap')
args.add_argument('-e','--epoch', default=5, type=int, help='Maximum epoch')
args = args.parse_args()
    
# use gpu
device = torch.device('cuda')

VggNet = Encoder(args.relulevel).to(device)
VggNet.train()
InvNet = Decoder(args.relulevel).to(device)
InvNet.load_state_dict(torch.load(f'./savemodel/InvNet_{args.epoch}_epoch.pth'))
InvNet.train()

# Since .png image has 4 channels RGBA, it need to be converted to RGB
content = Image.open(args.content).convert('RGB')
style = Image.open(args.style).convert('RGB')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
content = transform(content).unsqueeze(0).to(device)
style = transform(style).unsqueeze(0).to(device)

def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

with torch.no_grad():
    cf = VggNet(content)
    sf = VggNet(style)
    csf = style_swap(cf, sf, args.patchsize, 3)
    I_stylized = InvNet(csf)   
    I_stylized = denorm(I_stylized, device)

    save_image(I_stylized.cpu(), 
                os.path.join(args.output, (args.content.split('/')[-1].split('.')[0] + '_stylized_by_' + args.style.split('/')[-1])))