from __future__ import division
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from PIL import Image
import torchvision.utils as vutils
import argparse
import numpy as np
import matplotlib.pyplot as plt
from SDCGAN_sgmd import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size=64
batch_size=128


nz = 50 # latent vector的大小
ngf = 64 # generator feature map size
ndf = 64 # discriminator feature map size
nc = 3 # color channels


# Now, we can instantiate the generator and apply the weights_init function. Check out the printed model to see how the generator object is structured.

# Create the generator
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
netG = Generator(nz, ngf, nc).to(device)

fixed_noise = torch.randn(64, nz, device=device)

netG.load_state_dict(torch.load('sgan/chinese_ink.tar'))

with torch.no_grad():
    fake0 = netG(fixed_noise)[0].detach().cpu()
    fake = netG(fixed_noise)[1].detach().cpu()
# fake

# Plot the real images
plt.figure(figsize=(30,30))
plt.subplot(1,2,1)
plt.axis=("off")
plt.title("Real Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:32], padding=5, normalize=True).cpu(),(1,2,0)))
plt.imshow(np.transpose(vutils.make_grid(fake0, padding=2, normalize=True), (1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis=("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1,2,0)))

plt.show()

torchvision.utils.save_image(fake0, '3bts_f0.png')
torchvision.utils.save_image(fake, '3bts_f1.png')