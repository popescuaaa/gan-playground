import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from Generator import Generator
from Discriminator import Discriminator

import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt


class GAN:
    def __init__(self):
        torch.random.manual_seed(42)
        self.dim_latent_g = 128
        self.dim_output_g = 28 * 28
        self.dist_hidden = 128
        self.dim_output_d = 1
        self.dim_input_d = 28 * 28
        self.batch_size = 64
        self.learning_rate = 0.004
        self.num_epochs = 32

        self.d_train_iter = 2  # This is possible to have a relation with TTUR
        self.g_train_iter = 1

        self.dist_latent = torch.distributions.normal.Normal(loc=0, scale=1)  # Gaussian(0, 1)
        self.g = Generator(self.g_dist_latent, self.cfg_g)
        self.d = Discriminator()

        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ]
        )

        self.dataset = datasets.MNIST(root="dataset/", transform=self.transforms, download=True)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # TTUR => different learning rates
        self.optimizer_G = optim.Adam(self.g.parameters(), lr=self.learning_rate / 2)
        self.optimizer_D = optim.Adam(self.d.parameters(), lr=self.learning_rate * 2)

        self.criterion = nn.BCEWithLogitsLoss()  # This changed a little bit some loss values

    # max log(D(x)) + log(1 - D(G(z)))
    def train_discriminator(self, r, n):
        fake_data = self.g(n)
        real_data = r

        d_real = self.d(real_data).view(-1)
        d_fake = self.d(fake_data).view(-1)

        loss_fake = self.criterion(d_fake, torch.zeros_like(d_fake)).cuda()
        loss_real = self.criterion(d_real, torch.ones_like(d_real)).cuda()

        loss = 0.5 * loss_fake + 0.5 * loss_real
        self.d.zero_grad()

        loss.backward()
        self.optimizer_D.step()

        return loss

    #  min log(1 - D(G(z))) <-> max log(D(G(z))
    def train_generator(self, n):
        fake_data = self.g(n)
        output = self.d(fake_data).view(-1)

        loss = self.criterion(output, torch.ones_like(output)).cuda()
        self.g.zero_grad()

        loss.backward()
        self.optimizer_G.step()

        return loss

    def train_system(self):
        for epoch in range(self.num_epochs):
            for batch_idx, (real, _) in enumerate(self.loader):

                real = real.view(-1, 784).cuda()  # reshape -1 => don't know how many rows but should
                batch_size = real.shape[0]
                noise = self.g.generate_visual_sample(batch_size).cuda()

                for _ in range(self.g_train_iter):
                    loss_g = self.train_generator(noise)

                for _ in range(self.d_train_iter):
                    loss_d = self.train_discriminator(real, noise)

                if batch_idx == 0:
                    print(
                        f"Epoch [{epoch}/{self.num_epochs}] Batch {batch_idx}/{len(self.loader)} \
                          Loss D: {loss_d:.4f}, loss G: {loss_g:.4f}"
                    )
