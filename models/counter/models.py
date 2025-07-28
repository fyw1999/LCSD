import torch
import torch.nn as nn
from models.counter.layers import convDU, convLR
from masksembles.torch import Masksembles2D
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torchvision import models

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, features, args):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer_feat = [256, 'MASK', 128]

        self.reg_layer = make_layers(self.reg_layer_feat, in_channels = 512)
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        mu = self.density_layer(x)
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed

def make_layers(cfg, in_channels = 3, batch_norm = False, dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MASK':
            layers += [Masksembles2D(in_channels, 4, 2.0)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19(args):
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), args)
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model