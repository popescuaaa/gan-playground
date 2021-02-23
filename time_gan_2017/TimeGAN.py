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

        self.g = LSTMGenerator(self.g_dim_latent, self.g_dim_output, self.g_num_layers, self.g_dim_hidden)
        self.d = LSTMDiscriminator(self.d_dim_input, self.d_num_layers, self.d_dim_hidden)

        # Data
        self.ds = StockDataset('./csv/stock_data.csv', seq_len=10, config='Close')
        self.dl = DataLoader(self.ds, self.batch_size, shuffle=False, num_workers=10)

        # Optimizer
        self.d_optimizer = optim.Adam(list(self.g.parameters()), lr=self.learning_rate / 2)
        self.g_optimizer = optim.Adam(list(self.d.parameters()), lr=self.learning_rate * 2)

        self.criterion = nn.BCEWithLogitsLoss()

    def train_discriminator(self, data, dt, noise_data):

        self.d_optimizer.zero_grad()
        self.d.train()
        self.d.requires_grad_(True)
        self.g.requires_grad_(False)

        fake = self.g(noise_data, dt)

        d_real = self.d(data, dt)
        d_fake = self.d(fake, dt)

        loss_fake = self.criterion(d_fake, torch.zeros_like(d_fake))
        loss_real = self.criterion(d_real, torch.ones_like(d_real))

        loss = 0.5 * loss_fake + 0.5 * loss_real

        loss.backward()
        self.d_optimizer.step()

        return loss, fake

    def train_generator(self, noise_data, dt):
        self.g_optimizer.zero_grad()
        self.g.train()
        self.d.requires_grad_(False)
        self.g.requires_grad_(True)

        fake = self.g(noise_data, dt)
        output = self.d(fake, dt)

        loss = -1 * output.mean()

        loss.backward()
        self.g_optimizer.step()

        return loss

    def train_system(self):
        for epoch in range(self.num_epochs):
            for batch_idx, real in enumerate(self.dl):
                data, dt = real

                data = data.view(*data.shape, 1)
                data = data.float()

                dt = dt.view(*dt.shape, 1)
                dt = dt.float()

                noise_data = self.dist_latent.sample(sample_shape=(self.batch_size, self.seq_len, self.g_dim_latent))

                loss_g = self.train_generator(noise_data, dt)
                loss_d, fake = self.train_discriminator(data, dt, noise_data)

                if batch_idx == 0:
                    print(
                        f"Epoch [{epoch}/{self.num_epochs}] Batch {batch_idx}/{len(self.dl)} \
                          Loss D: {loss_d:.4f}, loss G: {loss_g:.4f}"
                    )

                    # wandb.log({
                    #     'epoch': epoch,
                    #     'd loss': loss_d,
                    #     'g loss': loss_g,
                    #     'Conditional on deltas fake sample': plot_time_series(
                    #         fake[0].view(-1).detach().cpu().numpy(),
                    #         '[Conditional (on deltas)] Fake sample'),
                    #     'Real sample': plot_time_series(
                    #         data[0].view(-1).cpu().numpy(),
                    #         '[Conditional (on deltas)] Real sample')
                    # })


def system_test(tgan: TimeGAN) -> None:
    noise_data = tgan.dist_latent.sample(sample_shape=(tgan.batch_size, tgan.seq_len, tgan.g_dim_latent))
    noise_data = noise_data.cuda()

    # Sample dt from dataset
    dt = tgan.ds.sample_dt()

    fake = tgan.g(noise_data, dt)
    fakes = [fake.view(-1).detach().cpu()]

    for _ in range(9):
        noise_data = tgan.dist_latent.sample(sample_shape=(tgan.batch_size, tgan.seq_len, tgan.g_dim_latent))
        noise_data = noise_data.cuda()

        noise_deltas = tgan.dist_latent.sample(sample_shape=(tgan.batch_size, tgan.seq_len, tgan.g_dim_latent))
        noise_deltas = noise_deltas.cuda()

        fakes.append(tgan.g(noise_data, noise_deltas).view(-1).detach().cpu())

    fake_mean = sum(fakes) / len(fakes)

    wandb.log({
        'Generated sample for RCGAN': plot_time_series(fake_mean.numpy(), '-')
    })


if __name__ == '__main__':
    # setup
    torch.random.manual_seed(42)

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    run_name = str(config.values())

    # wandb.init(config=config, project='time-gan-2017', name=run_name)

    time_gan = TimeGAN(config)
    time_gan.train_system()

    # test_generator(time_gan)
