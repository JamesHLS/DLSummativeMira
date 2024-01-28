# %% [markdown]
# **Main imports**

# %%
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
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

transform = transforms.Compose([
    transforms.RandomCrop(size=(32, 32), padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=True, download=True, transform=transform),
    batch_size=64, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=64, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset)}')
print(f'> Size of test dataset {len(test_loader.dataset)}')

# %% [markdown]
# **View some of the test dataset**

# %%
plt.rcParams['figure.dpi'] = 70
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    img = test_loader.dataset[i][0].numpy().transpose(1, 2, 0)
    plt.imshow(img)
    plt.xlabel(class_names[test_loader.dataset[i][1]])
plt.show()

# %% [markdown]
# **Define a simple model**

# %%



class SimpleResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SimpleResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(SimpleResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)



        self.linear = nn.Linear(32, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)



        out = torch.nn.functional.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    




N = SimpleResNet(block=SimpleResNetBlock, layers=[2, 2]).to(device)

# print the number of parameters - this should be included in your report
print(f'> Number of parameters {len(torch.nn.utils.parameters_to_vector(N.parameters()))}')

if len(torch.nn.utils.parameters_to_vector(N.parameters())) > 100000:
    print("> Warning: you have gone over your parameter budget and will have a grade penalty!")

# initialise the optimiser
optimiser = torch.optim.Adam(N.parameters(), lr=0.001)
plot_data = []
steps = 0

# %% [markdown]
# **Main training and testing loop**


# %%
import optuna


def train_model(N, optimiser, train_iterator, device, steps=5000):
    # keep within our optimisation step budget
    current_step = 0
    while (current_step < steps):
        # arrays for metrics
        train_loss_arr = np.zeros(0)
        train_acc_arr = np.zeros(0)

        # iterate through some of the train dataset
        for i in range(1000):
            x,t = next(train_iterator)
            x,t = x.to(device), t.to(device)

            optimiser.zero_grad()
            p = N(x)
            pred = p.argmax(dim=1, keepdim=True)
            loss = torch.nn.functional.cross_entropy(p, t)
            loss.backward()
            optimiser.step()
            current_step += 1

            train_loss_arr = np.append(train_loss_arr, loss.cpu().data)
            train_acc_arr = np.append(train_acc_arr, pred.data.eq(t.view_as(pred)).float().mean().item())

            if current_step >= steps:
                break

    return train_loss_arr.mean(), train_acc_arr.mean()

def validate_model(N, test_loader, device):
    N.eval()  # Set the model to evaluation mode
    test_acc_arr = np.zeros(0)

    with torch.no_grad():  # Do not calculate gradients to save memory
        for x, t in test_loader:
            x, t = x.to(device), t.to(device)
            p = N(x)
            pred = p.argmax(dim=1, keepdim=True)
            test_acc_arr = np.append(test_acc_arr, pred.data.eq(t.view_as(pred)).float().mean().item())

    return test_acc_arr.mean()
    

def objective(trial):
    # Define hyperparameters using the trial object
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    num_layers = 2
    num_blocks = [trial.suggest_int('num_blocks_layer_{}'.format(i), 1, 3) for i in range(num_layers)]
    
    # Create the model
    N = SimpleResNet(block=SimpleResNetBlock, layers=num_blocks).to(device)
    
    # Create the optimizer
    optimiser = torch.optim.Adam(N.parameters(), lr=lr)
    
    # Train the model and return the validation loss
    train_model(N, optimiser, train_iterator, device)  # You'll need to define this function
    test_acc = validate_model(N, test_loader=test_loader, device=device)  # And this one too
    return test_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print('Best trial:')
trial = study.best_trial
print('  Value: ', trial.value)
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))


# %% [markdown]
# **Inference on dataset**
# 
# This is useful for analysis but is entirely optional



