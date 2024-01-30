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

class_names = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','computer_keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm',]

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
    batch_size=64, drop_last=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
    batch_size=64, drop_last=True, num_workers=2)

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
# adapted from practical example https://github.com/atapour/dl-pytorch/blob/main/Conditional_GAN_Example/Conditional_GAN_Example.ipynb
class Generator(nn.Module):
    def __init__(self, latent_size=100, ksp=(4,2,1)):
        super(Generator, self).__init__()
        self.ksp = ksp
        self.n_channels = params['n_channels']
        self.fc = nn.Linear(latent_size, 128*7*7)  # Fully connected layer to reshape input
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=self.ksp[0], stride=self.ksp[1], padding=self.ksp[2]),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=self.ksp[0], stride=self.ksp[1], padding=self.ksp[2]),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, self.n_channels, kernel_size=self.ksp[0], stride=self.ksp[1], padding=self.ksp[2]),
        )

    def forward(self, x):
        x = self.fc(x)
        x = nn.ReLU(True)(x)
        x = x.view(-1, 128, 7, 7)  # Reshape to (batch_size, channels, height, width)
        x = self.layer(x)
        # x = F.interpolate(x, size=(32, 32))  
        return x
    
    def sample(self, z):
        x = self.forward(z)
        return x.view(x.size(0), self.n_channels, 32, 32)
    
    



# adapted from practical example https://github.com/atapour/dl-pytorch/blob/main/Conditional_GAN_Example/Conditional_GAN_Example.ipynb
class Discriminator(nn.Module):
    def __init__(self, params, ksp=(4,2,1)):
        super(Discriminator, self).__init__()
        self.ksp = ksp  # kernel size, stride, padding
        self.n_channels = params['n_channels']
        self.layer = nn.Sequential(
            # -> 3 channels 64x64
            nn.Conv2d(self.n_channels, 64, kernel_size=self.ksp[0], stride=self.ksp[1], padding=self.ksp[2]),
            # Leaky ReLU to prevent dying ReLU
            nn.LeakyReLU(0.2, inplace=True),  # tune negative slope
            # -> 64 channels 32x32
            nn.Conv2d(64, 128, kernel_size=self.ksp[0], stride=self.ksp[1], padding=self.ksp[2]),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # -> 128 channels 16x16
        )
    def forward(self, x):
        x = F.interpolate(x, size=(32, 32))
        x = self.layer(x)
        x = x.view(-1, 128*8*8)
        x = nn.Linear(128*8*8, 1)(x)
        x = nn.Sigmoid()(x)
        return x


# adapted from https://www.youtube.com/watch?v=_pIMdDWK5sc&t=1023s
class GAN(nn.Module):
    def __init__(self, latent_dim=100, gen_lr=0.0002, dis_lr=0.0002, D_ksp=(4,2,1), G_ksp=(4,2,1)):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.glr = gen_lr
        self.dlr = dis_lr
        self.generator = Generator(self.latent_dim, ksp=G_ksp)
        self.discriminator = Discriminator(params, ksp=D_ksp)
        self.valid = torch.randn(6, self.latent_dim)
        self.current_epoch = 0
        self.loss_g = 0
        self.loss_d = 0

    def forward(self, x):
        return self.generator(x)
    
    def get_loss(self, pred, target):
        return F.binary_cross_entropy(pred, target)
    
    def train(self, batch, batch_idx, optimizer_idx):
        reals = batch[0]
        noised = torch.randn(
            reals.shape[0], self.latent_dim
        )
        noised = noised.type_as(reals)

        # log(D(G(noised)))
        # train generator
        if optimizer_idx == 0:
            fakes = self(noised)   # generate fakes
            preds = self.discriminator(fakes)  # predict labels
            targets = torch.ones(reals.size(0), 1)
            targets = targets.type_as(reals)

            loss = self.get_loss(preds, targets)
            self.loss_g = loss
            return loss
        
        # log(D(real)) + log(1 - D(G(noised)))
        # train discriminator
        if optimizer_idx == 1:
            
            preds_real = self.discriminator(reals)
            targets_real = torch.ones(reals.size(0), 1)
            targets_real = targets_real.type_as(reals)
            loss = self.get_loss(preds_real, targets_real)

            preds_fake = self.discriminator(self(noised).detach())
            targets_fake = torch.zeros(reals.size(0), 1)
            targets_fake = targets_fake.type_as(reals)
            loss += self.get_loss(preds_fake, targets_fake)
            loss /= 2
            self.loss_d = loss
            return loss

    def config_optimizers(self):
        optimiserG = torch.optim.Adam(self.generator.parameters(), lr=self.glr)
        optimiserD = torch.optim.Adam(self.discriminator.parameters(), lr=self.dlr)
        return [optimiserG, optimiserD]

    def plot_imgs(self):
        z = self.valid.type_as(self.generator.fc.weight)
        samples = self(z).cpu()
        fig = plt.figure()
        for i in range(samples.size(0)):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(samples.detach()[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.title("Epoch {}".format(self.current_epoch))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
        plt.show()

    def on_epoch_end(self):
        self.plot_imgs()
        if self.current_epoch % 200:
          print('loss D: {:.3f}, loss G: {:.3f}'.format(self.loss_d, self.loss_g))
        self.current_epoch += 1




# hyperparameters
params = {
    'batch_size': train_loader.batch_size,
    'n_channels': 3,
    'n_latent': 7 # alters number of parameters
}

D = Discriminator(params).to(device)
G = Generator().to(device)
total_params = len(torch.nn.utils.parameters_to_vector(D.parameters())) + len(torch.nn.utils.parameters_to_vector(G.parameters())) 
print(f'> Number of model parameters {total_params}')
if total_params > 1000000:
    print("> Warning: you have gone over your parameter budget and will have a grade penalty!")


# initialise the optimiser
optimiserD = torch.optim.Adam(D.parameters(), lr=0.002)
optimiserG = torch.optim.Adam(G.parameters(), lr=0.002)

criterion = nn.BCELoss()

steps = 0
print("Crtierion: ", criterion)


# %%


# %%
model = GAN()
optimiserD, optimiserG = model.config_optimizers()
model.plot_imgs()
max_epochs = 50000
while (model.current_epoch < max_epochs):
    print("Step: ", model.current_epoch)

    # arrays for metrics
    logs = {}
    gen_loss_arr = np.zeros(0)
    dis_loss_arr = np.zeros(0)

    for i, batch in enumerate(train_loader):

        x, t = batch
        x, t = x.to(device), t.to(device)

        loss_g = model.train(batch, i, 0)
        loss_d = model.train(batch, i, 1)

        # update parameters
        optimiserD.zero_grad()
        loss_d.backward(retain_graph=True)
        optimiserD.step()

        optimiserG.zero_grad()
        loss_g.backward()
        optimiserG.step()

        model.on_epoch_end()

        gen_loss_arr = np.append(gen_loss_arr, loss_g.detach().numpy())
        dis_loss_arr = np.append(dis_loss_arr, loss_d.detach().numpy())

        if model.current_epoch >= max_epochs:
            break


    # sample model and visualise results (ensure your sampling code does not use x)
    G.eval()
    # g = G(torch.randn(x.size(0), 100).to(device))
    # samples = G.sample(g).cpu().detach()
    # if steps % 1000 == 0:
    print('loss D: {:.3f}, loss G: {:.3f}'.format(gen_loss_arr.mean(), dis_loss_arr.mean()))
    # plt.imshow(torchvision.utils.make_grid(g).cpu().data.clamp(0,1).permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    # plt.show()
    # plt.pause(0.0001)
    G.train()

    # update metrics
    logs['gen_loss'] = gen_loss_arr.mean()
    logs['dis_loss'] = dis_loss_arr.mean()

    
# %% [markdown]
# **Latent interpolations**


# %% [markdown]
# **FID scores**
# 
# Evaluate the FID from 10k of your model samples (do not sample more than this) and compare it against the 10k test images. Calculating FID is somewhat involved, so we use a library for it. It can take a few minutes to evaluate. Lower FID scores are better.

# %%
#%%capture
import os
from cleanfid import fid
from torchvision.utils import save_image

# %%
# define directories
real_images_dir = 'real_images'
generated_images_dir = 'generated_images'
num_samples = 10000 # do not change

# create/clean the directories
def setup_directory(directory):
    if os.path.exists(directory):
        os.removedirs(directory)
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

# %%
# compute FID
score = fid.compute_fid(real_images_dir, generated_images_dir, mode="clean")
print(f"FID score: {score}")


