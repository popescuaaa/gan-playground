import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

config = {
    'learning_rate': 0.1,

    'g': {
        'dim_latent': 128,  # z dimension for generator
        'dim_hidden': 128,  # size of the hidden layer
    },
}


class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        __module_list = [
            nn.Linear(cfg['dim_latent'], cfg['dim_hidden'], bias=True),
            nn.BatchNorm1d(cfg['dim_hidden'], affine=True, track_running_stats=True),
            nn.ReLU(),  # activation
            nn.Linear(cfg['dim_hidden'], cfg['dim_hidden'] * 2, bias=True),
            nn.BatchNorm1d(cfg['dim_hidden'] * 2, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(cfg['dim_hidden'] * 2, cfg['dim_hidden'] * 4, bias=True),
            nn.BatchNorm1d(cfg['dim_hidden'] * 4, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(cfg['dim_hidden'] * 4, 28 * 28, bias=True),  # size of a mnist sample
        ]

        # like a running container => in sequence
        self.__net = nn.Sequential(*__module_list)

    def forward(self, x):
        return self.__net(x)

class Discriminator(nn.Module):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        __module_list = [
            nn.Linear(dim_input, dim_input // 2, bias=True),
            nn.BatchNorm1d(dim_input // 2, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(dim_input // 2, dim_input // 4, bias=True),
            nn.BatchNorm1d(dim_input // 4, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(dim_input // 4, dim_input // 8, bias=True),
            nn.BatchNorm1d(dim_input // 8, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(dim_input // 8, dim_output, bias=True)

            # nn.Sigmoid()
            # Must be sigmoid or something that returns a value in [0, 1] to
            # make the BCELoss work (this was changed)
        ]

        self.__net = nn.Sequential(*__module_list)

    def forward(self, x):
        return self.__net(x)


env_learning_rate = config['learning_rate']

g_cfg = config['g']

g = Generator(g_cfg).cuda()
d = Discriminator(28 * 28, 1).cuda()  # we must generate a score

batch_size = 32
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# TTUR => different learning rates
optimizer_G = optim.Adam(g.parameters(), lr=config['learning_rate'] / 2)
optimizer_D = optim.Adam(d.parameters(), lr=config['learning_rate'] * 2)

criterion = nn.BCEWithLogitsLoss()  # This changed a little bit some loss values


# max log(D(x)) + log(1 - D(G(z)))
def trainD(real, noise):
    fake_data = g(noise)
    real_data = real

    d_real = d(real_data).view(-1)
    d_fake = d(fake_data).view(-1)

    loss_fake = criterion(d_fake, torch.zeros_like(d_fake)).cuda()
    loss_real = criterion(d_real, torch.ones_like(d_real)).cuda()

    loss = 0.5 * loss_fake + 0.5 * loss_real
    d.zero_grad()

    loss.backward()
    optimizer_D.step()

    return loss


#  min log(1 - D(G(z))) <-> max log(D(G(z))
def trainG(noise):
    fake_data = g(noise)
    output = d(fake_data).view(-1)

    loss = criterion(output, torch.ones_like(output)).cuda()
    g.zero_grad()

    loss.backward()
    optimizer_G.step()

    return loss


def train_system(noise, real):
    d_steps = 10
    g_steps = 10

    for _ in range(g_steps):
        loss_g = trainG(noise)

    for _ in range(d_steps):
        loss_d = trainD(real, noise)

    return loss_d, loss_g

num_epochs = 32
torch.random.manual_seed(42)

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):

        real = real.view(-1, 784).cuda()  # reshape -1 => don't know how many rows but should
        # be 784 columns
        batch_size = real.shape[0]

        noise = torch.randn(batch_size, config['g']['dim_latent']).cuda()

        lossD, lossG = train_system(noise, real)

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )
