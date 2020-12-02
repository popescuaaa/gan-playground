import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Generator import Generator
from Discriminator import Discriminator
from utils import *

import matplotlib.pyplot as plt


class GAN:
    def __init__(self):
        torch.random.manual_seed(42)
        self.dim_latent_g = 128
        self.dim_output_g = 28 * 28
        self.dim_hidden_g = 128
        self.dim_output_d = 1
        self.dim_input_d = 28 * 28
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.display_freq = 50

        self.d_train_iter = 2  # This is possible to have a relation with TTsUR
        self.g_train_iter = 1

        self.dist_latent = torch.distributions.normal.Normal(loc=0, scale=1)  # Gaussian(0, 1)
        self.g = Generator(self.dist_latent, self.dim_latent_g, self.dim_hidden_g, self.dim_output_g).cuda()
        self.d = Discriminator(self.dim_input_d, self.dim_output_d).cuda()

        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ]
        )

        self.dataset = datasets.MNIST(root="dataset/", transform=self.transforms, download=True)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # TTsUR => different learning rates
        self.optimizer_G = optim.Adam(self.g.parameters(), lr=self.learning_rate / 2)
        self.optimizer_D = optim.Adam(self.d.parameters(), lr=self.learning_rate * 2)

        self.criterion = nn.BCEWithLogitsLoss()  # This changed a little bit some loss values

    # max log(D(x)) + log(1 - D(G(z)))
    # r = real data
    # n = noise
    # l = labels
    def train_discriminator(self, r, n, l):
        fake_data = self.g(n, l)
        real_data = r

        d_real = self.d(real_data, l).view(-1)
        d_fake = self.d(fake_data, l).view(-1)

        loss_fake = self.criterion(d_fake, torch.zeros_like(d_fake)).cuda()
        loss_real = self.criterion(d_real, torch.ones_like(d_real)).cuda()

        loss = 0.5 * loss_fake + 0.5 * loss_real
        self.d.zero_grad()

        loss.backward()
        self.optimizer_D.step()

        return loss

    #  min log(1 - D(G(z))) <-> max log(D(G(z))
    # n = noise
    # l = labels
    def train_generator(self, n, l):
        fake_data = self.g(n, l)
        output = self.d(fake_data, l).view(-1)

        loss = self.criterion(output, torch.ones_like(output)).cuda()
        self.g.zero_grad()

        loss.backward()
        self.optimizer_G.step()

        return loss

    def train_system(self):
        generator_losses = []
        discriminator_losses = []

        for epoch in range(self.num_epochs):
            for batch_idx, (real, labels) in enumerate(self.loader):
                real = real.view(-1, 784).cuda()
                labels = labels.cuda()
                batch_size = real.shape[0]
                noise = self.g.sample(batch_size).cuda()  # sample from the latent distribution of the Generator

                for _ in range(self.g_train_iter):
                    loss_g = self.train_generator(n=noise, l=labels)

                for _ in range(self.d_train_iter):
                    loss_d = self.train_discriminator(r=real, n=noise, l=labels)

                if batch_idx == 0:
                    print(
                        f"Epoch [{epoch}/{self.num_epochs}] Batch {batch_idx}/{len(self.loader)} \
                          Loss D: {loss_d:.4f}, loss G: {loss_g:.4f}"
                    )
                    generator_losses.append(loss_g)
                    discriminator_losses.append(loss_d)

                # if epoch % self.display_freq == 0:
                # fake_data = self.g.generate_visual_sample(self.batch_size, labels).detach()
                # logits = self.d(fake_data, labels)
                # fake_grid = create_grid_plot(fake_data, logits)
                # plt.show()

        fig, ax = plt.subplots()
        ax.plot([i for i in range(len(generator_losses))], generator_losses, color='b')
        ax.plot([j for j in range(len(discriminator_losses))], discriminator_losses, color='r')
        ax.grid(linestyle='-', linewidth='0.5')

        plt.show()
