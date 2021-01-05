from TGenerator import LSTMGenerator
from TDiscriminator import LSTMDiscriminator
from TDataset import StockDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import torch


class TimeGAN:
    def __init__(self):
        # All models data are passed trough a container with linear and
        self.d_num_layers = 1  # default
        self.d_dim_hidden = 256  # default
        self.d_dim_input = 1
        self.d_train_iter = 2

        self.g_num_layers = 1
        self.g_dim_hidden = 256
        self.g_dim_latent = 128
        self.g_dim_output = 1
        self.g_dist_latent = torch.distributions.normal.Normal(loc=0, scale=1)  # Gaussian(0, 1)
        self.g_train_iter = 2

        self.batch_size = 10
        self.learning_rate = 1e-4
        self.num_epochs = 100

        self.g = LSTMGenerator(self.g_dim_latent, self.g_dim_output, self.g_dist_latent, self.g_num_layers,
                               self.g_dim_hidden)

        self.d = LSTMDiscriminator(self.d_dim_input, self.d_num_layers, self.d_dim_hidden)

        # Data
        self.ds = StockDataset()
        self.dl = DataLoader(self.ds, self.batch_size, shuffle=True, num_workers=10)

        # Optimizer
        self.d_optimizer = optim.Adam(self.g.parameters(), lr=self.learning_rate / 2)
        self.g_optimizer = optim.Adam(self.d.parameters(), lr=self.learning_rate * 2)

        self.criterion = nn.BCEWithLogitsLoss()

    def train_discriminator(self, real, noise):
        self.d_optimizer.zero_grad()
        self.d.train()
        self.d.requires_grad_(True)
        self.g.requires_grad_(False)

        fake = self.g(noise)

        d_real = self.d(real).view(-1)
        d_fake = self.d(fake).view(-1)

        loss_fake = self.criterion(d_fake, torch.zeros_like(d_fake))
        loss_real = self.criterion(d_real, torch.ones_like(d_real))

        loss = 0.5 * loss_fake + 0.5 * loss_real

        loss.backward()
        self.d_optimizer.step()

        return loss

    def train_generator(self, noise):
        self.g_optimizer.zero_grad()
        self.d.eval()
        self.g.train()
        self.d.requires_grad_(False)
        self.g.requires_grad_(True)

        fake = self.g(noise)
        output = self.d(fake).view(-1)

        loss = -1 * output.mean()

        loss.backward()
        self.g_optimizer.step()

        return loss

    def train_system(self):
        for epoch in range(self.num_epochs):
            for batch_idx, real in enumerate(self.dl):

                real = real.view(-1, 1).cuda()
                batch_size = real.shape[0]
                noise = self.g.sample(batch_size).cuda()

                loss_g = self.train_generator(noise)

                loss_d = self.train_discriminator(real, noise)

                if batch_idx == 0:
                    print(
                        f"Epoch [{epoch}/{self.num_epochs}] Batch {batch_idx}/{len(self.dl)} \
                          Loss D: {loss_d:.4f}, loss G: {loss_g:.4f}"
                    )


def test_generator(g: LSTMGenerator) -> bool:
    pass


def test_discriminator(d: LSTMDiscriminator) -> bool:
    pass


if __name__ == '__main__':
    time_gan = TimeGAN()
    time_gan.train_system()
