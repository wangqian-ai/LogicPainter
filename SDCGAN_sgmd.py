from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.nz = nz-3
        self.ngf = ngf
        self.nc = nc
        self.fc1 = nn.Linear(self.nz, 4 * 4 * (ngf * 8))  # This seems.. wrong.  Should it be dim*8?
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.relu1 = nn.ReLU(True)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # torch.nn.ConvTranspose2d(in_channels, out_channels,
            # kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
            # nn.ConvTranspose2d( nz-3, ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        x = self.fc1(input[:,3:])
        x = x.view(-1, self.ngf * 8, 4, 4)
        x = self.relu1(self.bn1(x))
        x = self.main(x)
        darkness_mask = torch.mean(x, dim=1, keepdim=True)
        darkness_mask = 1. - darkness_mask
        # black = torch.zeros_like(input[:, :3])
        # color = black.view(-1, 3, 1, 1)
        # normalizer, _ = torch.max(darkness_mask, dim=2, keepdim=True)
        # normalizer, _ = torch.max(normalizer, dim=3, keepdim=True)  # max darkness_mask
        # darkness_mask = darkness_mask / normalizer
        # color = input[:, :3].view(-1, 3, 1, 1)
        # color = torch.where(color > 1, torch.sigmoid(color), color)
        # color = torch.where(color < 0, torch.sigmoid(color), color)
        # color = torch.abs(torch.tanh(input[:, :3].view(-1, 3, 1, 1)))
        # color = (torch.tanh(input[:, :3].view(-1, 3, 1, 1)) + 1.) / 2.
        color = torch.sigmoid(input[:, :3].view(-1, 3, 1, 1))
        # color = color.repeat(1, 1, 64, 64)
        # white = torch.ones_like(color)
        # color_r = input[:, :1].view(-1, 1, 1, 1)
        # color_g = input[:, 1:2].view(-1, 1, 1, 1)
        # color_b = input[:, 2:3].view(-1, 1, 1, 1)
        # color_r = torch.ones_like(input[:, :1].view(-1, 1, 1, 1))
        # color_g = torch.zeros_like(input[:, :1].view(-1, 1, 1, 1))
        # color_b = torch.zeros_like(input[:, :1].view(-1, 1, 1, 1))
        # color = torch.cat([color_r, color_g, color_b], 1)

        z = darkness_mask * color + (1. - darkness_mask)
        return x, z

class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        x = self.main(input)
        x = torch.sigmoid(x)
        return x
