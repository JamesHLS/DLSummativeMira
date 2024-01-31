# -*- coding: utf-8 -*-
"""example_cifar100_generative_model_redo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qWCQWwoThcSshTcRUr5ONQGI339zffWp

**Main imports**
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from IPython import display as disp
import optuna

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""**Import dataset**"""

# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x









learning_rate = 0.0002
z_dim = 60
batch_size = 64
image_size = 64
channels_img = 3
num_epochs = 10
# features_d = features_g
features_d = 15
features_g = 15
means = [0.5 for _ in range(channels_img)]
stds = [0.5 for _ in range(channels_img)]

invTrans = torchvision.transforms.Compose([ torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                torchvision.transforms.Normalize(mean = [ -0.5, -0.5, -0.5],
                                                     std = [ 1., 1., 1. ]),
                               ])

# inv_tensor = invTrans(inp_tensor)

params = {
    'batch_size': batch_size,
    'n_channels': channels_img,
    'n_latent': z_dim # alters number of parameters
}

class_names = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','computer_keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm',]
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=means,
        std=stds
    )
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=True, download=True, transform=transforms),
    batch_size=batch_size, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=means,
            std=stds)])),
    batch_size=batch_size, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset)}')
print(f'> Size of test dataset {len(test_loader.dataset)}')


"""**This is an autoencoder pretending to be a generative model**"""

# DC-GAN
"""
https://arxiv.org/abs/1511.06434
generator:
100-dim vector z -convtranspose2d upscale-> 1024channels, 4x4 -> 512 channels, 8x8 -> 256 channels, 16x16 -> 128 channels, 32x32 -> 3 channels, 64x64
no fully connected layers

discriminator:
3 channels, 64x64 -> 128 channels, 32x32 -> 256 channels, 16x16 -> 512 channels, 8x8 -> 1024 channels, 4x4 -> 1 channel, 1x1

architecture guidelines from paper:
- use batchnorm in both generator and discriminator
- use ReLU in generator for all layers except for output which uses Tanh
- use LeakyReLU in discriminator for all layers
- remove fully connected hidden layers for deeper architectures
- replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator)


- batchsize 128
- weights intialised from normal distribution with mean 0, stdev 0.02
- slope of leak 0.2
- lr  0.0002, momentum 0.5 (stablised training)


implementation adapted from https://www.youtube.com/watch?v=IZtv9s_Wx9I&list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va&index=3

"""

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, dropout=0.0):
        super(Discriminator, self).__init__()
        self.dropout = dropout
        self.disc = nn.Sequential(
            # paper: first layer of disc no batchnorm
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1), # 32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), # 16x16
            self._block(features_d*2, features_d*4, 4, 2, 1), # 8x8
            self._block(features_d*4, features_d*8, 4, 2, 1), # 4x4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0), # 1x1, fake or real
            nn.Dropout(dropout),
            nn.Sigmoid(),  # val 0-1
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            # batchnorm, bias false
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout),
        )
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    # z_dim: noise dimension
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # input: N x z_dim x 1 x 1
            # batchsize x noise dimension x 1 x 1
            self._block(z_dim, features_g*16, 4, 1, 0), # N x feature_g*16 x 4x4
            self._block(features_g*16, features_g*8, 4, 2, 1), # N x feature_g*8 x 8x8
            self._block(features_g*8, features_g*4, 4, 2, 1), # N x feature_g*4 x 16x16
            self._block(features_g*4, features_g*2, 4, 2, 1), # N x feature_g*2 x 32x32
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1), # N x channels_img x 64x64
            nn.Tanh(), # normalise to -1 to 1
        )
    def _block(self, in_channels, out_channels, kernal_size, stride, padding):
        return nn.Sequential(
            # upscale: ConvTranspose2d
            nn.ConvTranspose2d(in_channels, out_channels, kernal_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),  # paper: ReLu for generator
        )
    def forward(self, x):
        return self.gen(x)

    def sample(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)  # reshape z to [batch_size, n_latent, 1, 1]
        z = self.gen(z)
        return z

def initialize_weights(model):
    # paper: mean 0, std 0.02
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, height, width = 8, 3, 64, 64
    x = torch.randn((N, in_channels, height, width))
    disc = Discriminator(in_channels, features_d=8)
    initialize_weights(disc)
    print(disc(x).shape)
    print((N, 1, 1, 1))
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"

    gen = Generator(z_dim, in_channels, features_g=8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, height, width), "Generator test failed"
    print("Tests passed")


test()

# optuna to tune learning rate(options: 0.0001, 0.0002, 0.0003, 0.0004, 0.0005) 
            # z_dim(options: 40, 50, 60)
            # features_d (range: 10-15)
            # features_g (range: 10-15)
            # num_epochs (range: 5-60)
# MINIMIZE FID SCORE

def objective(trial):
    features_d = 15
    features_g = 15
    learning_rate = trial.suggest_categorical("learning_rate", [0.0002, 0.0005])
    num_epochs = trial.suggest_int("num_epochs", 25, 40)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    params = {
    'batch_size': batch_size,
    'n_channels': channels_img,
    'n_latent': z_dim # alters number of parameters
}
    D = Discriminator(channels_img, features_d, dropout=dropout).to(device)
    G = Generator(z_dim, channels_img, features_g).to(device)

    G_params = G.parameters()
    D_params = D.parameters()

    print(f'> Number of discriminator parameters {len(torch.nn.utils.parameters_to_vector(D.parameters()))}'
            f' and generator parameters {len(torch.nn.utils.parameters_to_vector(G.parameters()))}')
    total_params = len(torch.nn.utils.parameters_to_vector(D.parameters())) + len(torch.nn.utils.parameters_to_vector(G.parameters()))
    print('> Number of total parameters', total_params)
    if total_params > 1000000:
        print("> Warning: you have gone over your parameter budget and will have a grade penalty!")

    # Commented out IPython magic to ensure Python compatibility.
    # %load_ext tensorboard


    G = Generator(z_dim, channels_img, features_g).to(device)
    D = Discriminator(channels_img, features_d, dropout=dropout).to(device)
    initialize_weights(G)
    initialize_weights(D)
    # paper: betas 0.5, 0.999
    optimiser_G = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimiser_D = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, z_dim, 1, 1).to(device)
    step = 0
    max_steps = 50000

    G.train()
    D.train()

    d_losses = []
    g_losses = []

    # Commented out IPython magic to ensure Python compatibility.
    # Training Loop


    for epoch in range(num_epochs):
        if step >= max_steps:
            break
        for batch_idx, (real, _) in enumerate(train_loader):
            real = real.to(device)
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = G(noise)
            # discriminator
            # D(x): real data, D(G(z)): fake data
            # log(D(x)) + log(1 - D(G(z)))
            disc_real = D(real).reshape(-1)
            loss_D_real = criterion(disc_real, torch.ones_like(disc_real))
            D_x = disc_real.mean().item()
            disc_fake = D(fake).reshape(-1)
            loss_D_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_D = (loss_D_real + loss_D_fake)/2
            D.zero_grad()
            loss_D.backward(retain_graph=True)
            D_G_z1 = disc_fake.mean().item()
            optimiser_D.step()

            # generator
            # G(z): generated data, D(G(z)): disc output on generated data
            # max log(D(G(z)))
            output = D(fake).reshape(-1)
            loss_G = criterion(output, torch.ones_like(output))
            G.zero_grad()
            loss_G.backward()
            D_G_z2 = output.mean().item()
            optimiser_G.step()



            if batch_idx % 200 == 0:
                d_losses.append(loss_D.item())
                g_losses.append(loss_G.item())
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                       % (epoch, num_epochs, batch_idx, len(train_loader),
                        loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))
            step += 1
            if step >= max_steps:
                break


    z = torch.randn(64, z_dim, 1, 1).to(device)

    d_loss_mean = np.mean(d_losses)
    g_loss_mean = np.mean(g_losses)


    """**FID scores**

    Evaluate the FID from 10k of your model samples (do not sample more than this) and compare it against the 10k test images. Calculating FID is somewhat involved, so we use a library for it. It can take a few minutes to evaluate. Lower FID scores are better.
    """


    # Commented out IPython magic to ensure Python compatibility.
    import shutil
    import os
    from cleanfid import fid
    from torchvision.utils import save_image

    # define directories
    real_images_dir = 'real_images_dropout'
    generated_images_dir = 'generated_images_dropout'
    num_samples = 10000 # do not change

    # create/clean the directories
    def setup_directory(directory):
        if os.path.exists(directory):
            shutil.rmtree(directory) # remove any existing (old) data
        os.makedirs(directory)

    setup_directory(real_images_dir)
    setup_directory(generated_images_dir)

    # generate and save 10k model samples
    num_generated = 0
    while num_generated < num_samples:

        # sample from your model, you can modify this
        z = torch.randn(params['batch_size'], params['n_latent']).to(device)
        samples_batch = G.sample(z).cpu().detach()

        for image in samples_batch:
            if num_generated >= num_samples:
                break
            save_image(image, os.path.join(generated_images_dir, f"gen_img_{num_generated}.png"))
            num_generated += 1

    # save 10k images from the CIFAR-100 test dataset
    num_saved_real = 0
    while num_saved_real < num_samples:
        real_samples_batch, _ = next(test_iterator)
        for image in real_samples_batch:
            if num_saved_real >= num_samples:
                break
            save_image(image, os.path.join(real_images_dir, f"real_img_{num_saved_real}.png"))
            num_saved_real += 1

    # compute FID
    score = fid.compute_fid(real_images_dir, generated_images_dir, mode="clean")
    print(f"FID score: {score}")

    return score


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)