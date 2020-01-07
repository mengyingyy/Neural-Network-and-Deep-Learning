# This model is used as a controlled group
# usage: python train.py -c ./content -s ./style
# modified from https://github.com/Joanhxp/Style-Swap/blob/master/main.py

import os
import argparse
import torch
import glob
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from model import *

args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-c','--content', default="./content", type=str, help='Content dataset folder path')
args.add_argument('-s','--style', default="./style", type=str, help='Style dataset folder path')
args.add_argument('-p','--patchsize', default=3, type=int, help='Patch size of style swap')
args.add_argument('-r','--relulevel', default=3, type=int, help='Relu-level of style swap')
args.add_argument('-e','--epoch', default=5, type=int, help='Maximum epoch')
args.add_argument('-b','--batch', default=4, type=int, help='Batch size of training')
args.add_argument('-tv','--tvweight', default=1e-5, type=float, help='TV weight of regularization')
args.add_argument('-lr','--learningrate', default=0.001, type=float, help='Learning rate')
args = args.parse_args()

class PreprocessDataset(Dataset):
    def __init__(self, content_dir, style_dir, train_trans):
        content_images = glob.glob((content_dir + '/*'))
        np.random.shuffle(content_images)
        style_images = glob.glob((style_dir + '/*'))
        np.random.shuffle(style_images)
        self.images_pairs = list(zip(content_images, style_images))
        self.transforms = train_trans

    def __len__(self):
        return len(self.images_pairs)

    def __getitem__(self, index):
        content_name, style_name = self.images_pairs[index]
        content_image = Image.open(content_name).convert('RGB')
        style_image = Image.open(style_name).convert('RGB')
        if self.transforms:
            content_image = self.transforms(content_image)
            style_image = self.transforms(style_image)
        return {'c_img': content_image, 'c_name': content_name, 's_img': style_image, 's_name': style_name}

def TVLoss(img, tv_weight):
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss

device = torch.device('cuda')
    
VggNet = Encoder(args.relulevel).to(device)
VggNet.train()
InvNet = Decoder(args.relulevel).to(device)
InvNet.train()
    
train_trans = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
train_dataset = PreprocessDataset(args.content, args.style, train_trans)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
optimizer = torch.optim.Adam(InvNet.parameters(), lr=args.learningrate)
criterion = nn.MSELoss()

loss_list = []
i = 0
for epoch in range(1, args.epoch+1):
    for _, image in enumerate(train_dataloader):
        content = image['c_img'].to(device)
        style = image['s_img'].to(device)
        # encode
        cf = VggNet(content)
        sf = VggNet(style)
        # style swap
        csf = style_swap(cf, sf, args.patchsize, stride=3)
        # decode
        I_stylized = InvNet(csf)
        I_c = InvNet(cf)
        I_s = InvNet(sf)
        # put the output back into encoder to computer loss
        P_stylized = VggNet(I_stylized)     # size: 2 x 256 x 64 x 64
        P_c = VggNet(I_c)
        P_s = VggNet(I_s)
        # computer total loss
        loss_stylized = criterion(P_stylized, csf) + criterion(P_c, cf) + criterion(P_s, sf)
        loss_tv = TVLoss(I_stylized, args.tvweight)
        loss = loss_stylized + loss_tv
        loss_list.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
        print("%d / %d epoch\tloss: %.4f\tloss_stylized: %.4f loss_tv: %.4f" % (epoch, args.epoch, loss.item()/args.batch, loss_stylized.item()/args.batch, loss_tv.item()/args.batch))
        
    torch.save(InvNet.state_dict(), f'./savemodel/InvNet_{epoch}_epoch.pth')

with open('loss.txt', 'w') as f:
    for l in loss_list:
        f.write(f'{l}\n')