# -*- coding: utf-8 -*-
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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    image_size=64
    batch_size=8
    # dataroot="/home/gc/wq/neural-painters-pytorch/notebooks/strokes_dataset"
    dataroot = "stroke_oilp"
    num_workers = 2
    dataset = torchvision.datasets.ImageFolder(root=dataroot, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    real_batch=next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis=("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
    print(len(dataset))
    print(dataset[0]) # 训练集第一张图像张量以及对应的标签，二维元组
    print(dataset[0][0]) # 训练集第一张图像的张量
    print(dataset[0][1]) # 训练集第一张图像的标签

    nz = 50 # latent vector的大小
    ngf = 64 # generator feature map size
    ndf = 64 # discriminator feature map size
    nc = 3 # color channels


    # Now, we can instantiate the generator and apply the weights_init function. Check out the printed model to see how the generator object is structured.

    # Create the generator
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    netG = Generator(nz, ngf, nc).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Now, as with the generator, we can create the discriminator, apply the weights_init function, and print the model’s structure.

    # Create the Discriminator
    netD = Discriminator(ndf, nc).to(device)


    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    lr = 0.0002
    beta1 = 0.5

    loss_fn = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, device=device)
    d_optimizer = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    g_optimizer = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    num_epochs = 50
    G_losses = []
    D_losses = []

    f = open('./sgan/sganloss.txt', 'w')

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            # 训练discriminator, maximize log(D(x)) + log(1-D(G(z)))

            # 首先训练真实图片
            netD.zero_grad()

            real_images = data[0].to(device)
            b_size = real_images.size(0)
            label = torch.ones(b_size).to(device)
            output = netD(real_images).view(-1)

            real_loss = loss_fn(output, label)
            real_loss.backward()
            D_x = output.mean().item()

            # 然后训练生成的假图片
            noise = torch.randn(b_size, nz, device=device)
            fake_images, fake_images2 = netG(noise)
            label.fill_(0)
            output = netD(fake_images.detach()).view(-1)
            fake_loss = loss_fn(output, label)
            fake_loss.backward()
            D_G_z1 = output.mean().item()
            loss_D = real_loss + fake_loss
            d_optimizer.step()

            # 训练Generator
            netG.zero_grad()
            label.fill_(1)
            output = netD(fake_images).view(-1)
            loss_G = loss_fn(output, label)
            loss_G.backward()
            D_G_z2 = output.mean().item()
            g_optimizer.step()


            f.write("[{}/{}] [{}/{}] Loss_D: {:.4f} Loss_G {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}\n"
                      .format(epoch, num_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))

            if i % 50 == 0:
                print("[{}/{}] [{}/{}] Loss_D: {:.4f} Loss_G {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}"
                      .format(epoch, num_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())

    f.close()

    # for i, data in enumerate(dataloader):
    #     # 训练discriminator, maximize log(D(x)) + log(1-D(G(z)))
    #
    #     # 首先训练真实图片
    #     netD.zero_grad()
    #
    #     real_images = data[0].to(device)
    #     b_size = real_images.size(0)
    #     label = torch.ones(b_size).to(device)
    #     output = netD(real_images).view(-1)
    #
    #     real_loss = loss_fn(output, label)
    #     real_loss.backward()
    #     D_x = output.mean().item()
    #
    #     # 然后训练生成的假图片
    #     noise = torch.randn(b_size, nz, device=device)
    #     fake_images, fake_images2 = netG(noise)
    #     label.fill_(0)
    #     output = netD(fake_images2.detach()).view(-1)
    #     fake_loss = loss_fn(output, label)
    #     fake_loss.backward()
    #     D_G_z1 = output.mean().item()
    #     loss_D = real_loss + fake_loss
    #     d_optimizer.step()
    #
    #     # 训练Generator
    #     netG.zero_grad()
    #     label.fill_(1)
    #     output = netD(fake_images).view(-1)
    #     loss_G = loss_fn(output, label)
    #     loss_G.backward()
    #     D_G_z2 = output.mean().item()
    #     g_optimizer.step()
    #
    #     if i % 50 == 0:
    #         print("[{}/{}] [{}/{}] Loss_D: {:.4f} Loss_G {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}"
    #           .format(epoch, num_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))
    #
    #     G_losses.append(loss_G.item())
    #     D_losses.append(loss_D.item())

    with torch.no_grad():
        fake0 = netG(fixed_noise)[0].detach().cpu()
        fake = netG(fixed_noise)[1].detach().cpu()
    # fake

    real_batch = next(iter(dataloader))

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

    torch.save(netG.state_dict(), "sgan/oil_painting.tar")
