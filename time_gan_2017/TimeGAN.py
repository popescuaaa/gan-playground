from TGenerator import LSTMGenerator
from TDiscriminator import LSTMDiscriminator
from TDataset import StockDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

import wandb
import yaml

from plot import plot_time_series


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

        # Latent distribution
        self.dist_latent = torch.distributions.normal.Normal(loc=0, scale=1)  # Gaussian( 0, 1 )

        # System
        self.batch_size = int(cfg['system']['batch_size'])
        self.seq_len = int(cfg['system']['seq_len'])
        self.learning_rate = float(cfg['system']['learning_rate'])
        self.num_epochs = int(cfg['system']['num_epochs'])

        self.g = LSTMGenerator(self.g_dim_latent, self.g_dim_output, self.g_num_layers, self.g_dim_hidden).cuda()
        self.d = LSTMDiscriminator(self.d_dim_input, self.d_num_layers, self.d_dim_hidden).cuda()

        # Data
        self.ds = StockDataset(seq_len=self.seq_len, normalize=True)
        self.dl = DataLoader(self.ds, self.batch_size, shuffle=True, num_workers=10)

        # Optimizer
        self.d_optimizer = optim.Adam(list(self.g.parameters()), lr=self.learning_rate / 2)
        self.g_optimizer = optim.Adam(list(self.d.parameters()), lr=self.learning_rate * 2)

        self.criterion = nn.BCEWithLogitsLoss()

    def train_discriminator(self, real, noise):

        self.d_optimizer.zero_grad()
        self.d.train()
        self.d.requires_grad_(True)
        self.g.requires_grad_(False)

        fake = self.g(noise)

        d_real = self.d(real)
        d_fake = self.d(fake)

        loss_fake = self.criterion(d_fake, torch.zeros_like(d_fake))
        loss_real = self.criterion(d_real, torch.ones_like(d_real))

        loss = 0.5 * loss_fake + 0.5 * loss_real

        loss.backward()
        self.d_optimizer.step()

        return loss, fake

    def train_generator(self, noise):
        self.g_optimizer.zero_grad()
        self.g.train()
        self.d.requires_grad_(False)
        self.g.requires_grad_(True)

        fake = self.g(noise)
        output = self.d(fake)

        loss = -1 * output.mean()

        loss.backward()
        self.g_optimizer.step()

        return loss

    def train_system(self):
        for epoch in range(self.num_epochs):
            for batch_idx, real in enumerate(self.dl):
                #  real = real.view(self.seq_len, self.batch_size, 1)
                real = real.view(*real.shape, 1)
                real = real.cuda()

                noise = self.dist_latent.sample(sample_shape=(self.batch_size, self.seq_len, self.g_dim_latent))
                noise = noise.cuda()

                loss_g = self.train_generator(noise)
                loss_d, fake = self.train_discriminator(real, noise)

                if batch_idx == 0:
                    print(
                        f"Epoch [{epoch}/{self.num_epochs}] Batch {batch_idx}/{len(self.dl)} \
                          Loss D: {loss_d:.4f}, loss G: {loss_g:.4f}"
                    )

                    wandb.log({
                        'epoch': epoch,
                        'd loss': loss_d,
                        'g loss': loss_g,
                        'sample fig': plot_time_series(fake[0].view(-1).detach().cpu().numpy(), 'Fake '
                                                                                                'sample'),
                        'sample real': plot_time_series(real[0].view(-1).cpu().numpy(), 'Real sample')
                    })


def test_generator(tgan: TimeGAN) -> bool:
    noise = tgan.dist_latent.sample(sample_shape=(tgan.seq_len * tgan.batch_size, tgan.g_dim_latent)). \
        view(tgan.batch_size, tgan.seq_len, tgan.g_dim_latent)
    noise = noise.cuda()
    fake = tgan.g(noise)
    fake = fake.cpu()
    print(fake)

    return True


def test_discriminator(tgan: TimeGAN) -> bool:
    pass


if __name__ == '__main__':
    # setup
    torch.random.manual_seed(42)

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    run_name = str(config.values())

    wandb.init(config=config, project='time-gan-2017', name=run_name)

    time_gan = TimeGAN(config)
    time_gan.train_system()

    test_generator(time_gan)

'''
d -> 10 (2 * seq) -> plot
g -> face 10 rulari peste prima seq -> mean
plot -> cu mean si variance
'''
