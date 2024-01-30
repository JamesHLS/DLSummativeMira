# %% [markdown]
# **Main imports**

# %%
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from IPython import display as disp

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %% [markdown]
# **Import dataset**

# %%
# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

learning_rate = 0.0002
z_dim = 100
batch_size = 128
image_size = 64
channels_img = 3
num_epochs = 5
# features_d = features_g = 64
features_d = 64
features_g = 64
means = [0.5 for _ in range(channels_img)]
stds = [0.5 for _ in range(channels_img)]

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
    batch_size=64, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=means, 
            std=stds)])),
    batch_size=64, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset)}')
print(f'> Size of test dataset {len(test_loader.dataset)}')

# %% [markdown]
# **View some of the test dataset**

# %%
# let's view some of the training data
plt.rcParams['figure.dpi'] = 100
x,t = next(train_iterator)
x,t = x.to(device), t.to(device)
plt.imshow(torchvision.utils.make_grid(x).cpu().numpy().transpose(1, 2, 0), cmap=plt.cm.binary)
plt.show()

# %% [markdown]
# **This is an autoencoder pretending to be a generative model**

# %%
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

"""

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # paper: first layer of disc no batchnorm
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1), # 32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), # 16x16
            self._block(features_d*2, features_d*4, 4, 2, 1), # 8x8
            self._block(features_d*4, features_d*8, 4, 2, 1), # 4x4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0), # 1x1, fake or real
            nn.Sigmoid(),  # val 0-1
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            # batchnorm, bias false
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
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
    
def initialize_weights(model):
    # paper: mean 0, std 0.02
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, height, width = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, height, width))
    disc = Discriminator(in_channels, features_d=8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"

    gen = Generator(z_dim, in_channels, features_g=8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, height, width), "Generator test failed"
    print("Tests passed")


test()






# N = None

# print(f'> Number of model parameters {len(torch.nn.utils.parameters_to_vector(N.parameters()))}')
# if len(torch.nn.utils.parameters_to_vector(N.parameters())) > 1000000:
#     print("> Warning: you have gone over your parameter budget and will have a grade penalty!")

# # initialise the optimiser
# optimiser = torch.optim.Adam(N.parameters(), lr=0.001)
# steps = 0

# %%
# Training Loop
from torch.utils.tensorboard import SummaryWriter

G = Generator(z_dim, channels_img, features_g).to(device)
D = Discriminator(channels_img, features_d).to(device)
initialize_weights(G)
initialize_weights(D)
# paper: betas 0.5, 0.999
optimiser_G = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimiser_D = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, z_dim, 1, 1).to(device)
writer_real = SummaryWriter(f'logs/real')
writer_fake = SummaryWriter(f'logs/fake')
step = 0
max_steps = 10

G.train()
D.train()

while step < max_steps:
    for batch_idx, (real, _) in enumerate(train_loader):
        real = real.to(device)
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake = G(noise)
        # discriminator
        # D(x): real data, D(G(z)): fake data
        # log(D(x)) + log(1 - D(G(z)))
        disc_real = D(real).reshape(-1)
        loss_D_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = D(fake).reshape(-1)
        loss_D_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_D = (loss_D_real + loss_D_fake)/2
        D.zero_grad()
        loss_D.backward(retain_graph=True)
        optimiser_D.step()

        # generator
        # G(z): generated data, D(G(z)): disc output on generated data
        # max log(D(G(z)))
        output = D(fake).reshape(-1)
        loss_G = criterion(output, torch.ones_like(output))
        G.zero_grad()
        loss_G.backward()
        optimiser_G.step()

        # if batch_idx % 100 == 0:
        if true:
            print(f'Epoch [{step}/{max_steps}] Batch {batch_idx}/{len(train_loader)} \
                  Loss D: {loss_D:.4f}, loss G: {loss_G:.4f}')
            with torch.no_grad():
                fake = G(fixed_noise)  # generate with fixed noise
                # show samples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image('Real', img_grid_real, global_step=step)
                writer_fake.add_image('Fake', img_grid_fake, global_step=step)
        step += 1
        if step >= max_steps:
            break

























# %% [markdown]
# **Main training loop**




