import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

# modified from https://github.com/Joanhxp/Style-Swap/blob/master/model.py
decoder = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),    # :13
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 256, 3, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1),
        nn.ReLU(),    # :25
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 128, 3, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3, 1),
        nn.ReLU(),    # :32
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 64, 3, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU(),    # :39
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 3, 3, 1),
        )

class Encoder(nn.Module):
    def __init__(self, relu_level):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        relulevel_to_layer = {1:2, 2:7, 3:12, 4:21, 5:30}
        self.model = vgg[:relulevel_to_layer[relu_level]]
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, images):
        output = self.model(images)
        return output

class Decoder(nn.Module):
    def __init__(self, relu_level, decoder=decoder):
        super().__init__()
        decoder = list(decoder.children())
        relulevel_to_layer = {1:39, 2:32, 3:25, 4:13, 5:0}
        self.model = nn.Sequential(*decoder[relulevel_to_layer[relu_level]:])

    def forward(self, features):
        output = self.model(features)
        return output


# https://github.com/Joanhxp/Style-Swap/blob/master/style_swap.py
def style_swap(cf, sf, patch_size=3, stride=1):  # cf,sf  Batch_size x C x H x W
    b, c, h, w = sf.size()  # 2 x 256 x 64 x 64

    kh, kw = patch_size, patch_size
    sh, sw = stride, stride
    
    # Create convolutional filters by style features
    sf_unfold = sf.unfold(2, kh, sh).unfold(3, kw, sw)
    patches = sf_unfold.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(b, -1, c, kh, kw)
    patches_norm = torch.norm(patches.reshape(*patches.shape[:2], -1), dim=2).reshape(b, -1, 1, 1, 1)
    patches_norm = patches / patches_norm
    # patches size is 2 x 3844 x 256 x 3 x 3
    
    transconv_out = []
    for i in range(b):
        cf_temp = cf[i].unsqueeze(0)    # [1 x 256 x 64 x 64]
        patches_norm_temp = patches_norm[i]    # [3844, 256, 3, 3]
        patches_temp = patches[i]
        conv_out = F.conv2d(cf_temp, patches_norm_temp, stride=1)    # [1 x 3844 x 62, 62]
        
        one_hots = torch.zeros_like(conv_out)
        one_hots.scatter_(1, conv_out.argmax(dim=1, keepdim=True), 1)
        
        transconv_temp = F.conv_transpose2d(one_hots, patches_temp, stride=1)    # [1 x 256 x 64 x 64]
        overlap = F.conv_transpose2d(one_hots, torch.ones_like(patches_temp), stride=1)
        transconv_temp = transconv_temp / overlap
        transconv_out.append(transconv_temp)
    transconv_out = torch.cat(transconv_out, 0)    # [2 x 256 x 64 x 64]
    
    return transconv_out