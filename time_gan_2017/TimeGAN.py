from TGenerator import LSTMGenerator
from TDiscriminator import LSTMDiscriminator
from TDataset import StockDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from plot import plot_time_series
import wandb
import yaml


class TimeGAN:
    def __init__(self, cfg):
        # discriminator
        self.d_num_layers = int(cfg['d']['d_num_layers'])
        self.d_dim_hidden = int(cfg['d']['d_dim_hidden'])
        self.d_dim_input = int(cfg['d']['d_dim_input'])

        # generator
        self.g_num_layers = int(cfg['g']['g_num_layers'])
        self.g_dim_hidden = int(cfg['g']['g_dim_hidden'])
        self.g_dim_latent = int(cfg['g']['g_dim_latent'])
        self.g_dim_output = int(cfg['g']['g_dim_output'])
        self.g_dist_latent = torch.distributions.normal.Normal(loc=0, scale=1)  # Gaussian( 0, 1 )

        # system
        self.batch_size = int(cfg['system']['batch_size'])
        self.seq_len = int(cfg['system']['seq_len'])
        self.learning_rate = float(cfg['system']['learning_rate'])
        self.num_epochs = int(cfg['system']['num_epochs'])

        self.g = LSTMGenerator(self.g_dim_latent, self.g_dim_output, self.g_dist_latent, self.g_num_layers,
                               self.g_dim_hidden).cuda()

        self.d = LSTMDiscriminator(self.d_dim_input, self.d_num_layers, self.d_dim_hidden).cuda()

        # Data
        self.ds = StockDataset(seq_len=self.seq_len)
        self.dl = DataLoader(self.ds, self.batch_size, shuffle=False, num_workers=10)

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

                print(next(self.g.parameters()).is_cuda)
                batch_size = real.shape[0]
                real = real.view(self.seq_len, batch_size, 1)
                real = real.cuda()
                noise = self.g.sample(self.seq_len, batch_size).view(self.seq_len, batch_size, self.g_dim_latent)
                noise = noise.cuda()

                loss_g = self.train_generator(noise)
                loss_d = self.train_discriminator(real, noise)

                if batch_idx == 0:
                    print(
                        f"Epoch [{epoch}/{self.num_epochs}] Batch {batch_idx}/{len(self.dl)} \
                          Loss D: {loss_d:.4f}, loss G: {loss_g:.4f}"
                    )

                    wandb.log({'epoch': epoch, 'd loss': loss_d, 'g loss': loss_g})


    def test_system(self):
        pass


def test_generator(g: LSTMGenerator) -> bool:
    pass


def test_discriminator(d: LSTMDiscriminator) -> bool:
    pass


if __name__ == '__main__':
    # setup
    torch.random.manual_seed(42)

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    run_name = str(config.values())

    # wandb.init(config=config, project='time-gan-2017', name=run_name)

    time_gan = TimeGAN(config)
    time_gan.train_system()

'''
d -> 10 (2 * seq) -> plot
g -> face 10 rulari peste prima seq -> mean
plot -> cu mean si variance
'''
