from TGenerator import LSTMGenerator
from TDiscriminator import LSTMDiscriminator
from TDataset import StockDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import wandb
import yaml

import numpy as np

from plot import plot_time_series
from sklearn.metrics import mean_absolute_error
from metrics import visualization_metrics


class TimeGAN:
    def __init__(self, cfg):
        # cuda
        self.use_cuda = torch.torch.cuda.is_available()
        self.device = torch.device(cfg['system']['device'] if self.use_cuda else "cpu")

        # discriminator
        self.d_num_layers = int(cfg['d']['d_num_layers'])
        self.d_dim_hidden = int(cfg['d']['d_dim_hidden'])
        self.d_dim_input = int(cfg['d']['d_dim_input'])
        self.d_iter = int(cfg['d']['d_iter'])

        # generator
        self.g_num_layers = int(cfg['g']['g_num_layers'])
        self.g_dim_hidden = int(cfg['g']['g_dim_hidden'])
        self.g_dim_latent = int(cfg['g']['g_dim_latent'])
        self.g_dim_output = int(cfg['g']['g_dim_output'])
        self.g_iter = int(cfg['g']['g_iter'])

        # Latent distribution
        self.dist_latent = torch.distributions.normal.Normal(loc=0, scale=1)  # Gaussian( 0, 1 )

        # System
        self.batch_size = int(cfg['system']['batch_size'])
        self.seq_len = int(cfg['system']['seq_len'])
        self.learning_rate = float(cfg['system']['learning_rate'])
        self.num_epochs = int(cfg['system']['num_epochs'])

        self.g = LSTMGenerator(self.g_dim_latent,
                               self.g_dim_output,
                               self.g_num_layers,
                               self.g_dim_hidden).to(self.device)

        self.d = LSTMDiscriminator(self.d_dim_input,
                                   self.d_num_layers,
                                   self.d_dim_hidden).to(self.device)

        # Data
        self.ds = StockDataset('./csv/AAPL.csv', seq_len=self.seq_len, config='Close')
        self.dl = DataLoader(self.ds, self.batch_size, shuffle=True, num_workers=10)

        # Optimizer
        self.d_optimizer = optim.Adam(list(self.g.parameters()), lr=self.learning_rate / 2)
        self.g_optimizer = optim.Adam(list(self.d.parameters()), lr=self.learning_rate * 2)

        self.criterion = nn.BCEWithLogitsLoss()

    def train_discriminator(self, noise_stock, stock, dt, mean):
        self.d_optimizer.zero_grad()
        self.d.train()
        self.d.requires_grad_(True)
        self.g.requires_grad_(False)

        fake = self.g(noise_stock, dt, mean)

        d_real = self.d(stock, dt, mean)
        d_fake = self.d(fake, dt, mean)

        loss = -torch.log(torch.sigmoid(d_real - d_fake)).mean()
        loss.backward()
        self.d_optimizer.step()

        return loss, fake

    def train_generator(self, noise_stock, stock, dt, mean):
        self.g_optimizer.zero_grad()
        self.g.train()
        self.d.requires_grad_(False)
        self.g.requires_grad_(True)

        fake = self.g(noise_stock, dt, mean)
        d_fake = self.d(fake, dt, mean)
        d_real = self.d(stock, dt, mean)

        loss = -torch.log(torch.sigmoid(d_fake - d_real)).mean()
        loss.backward()
        self.g_optimizer.step()

        return loss, fake

    def generate_distributions(self):
        _noise = self.dist_latent.sample(sample_shape=(1, self.seq_len, self.g_dim_latent))
        _noise = _noise.to(self.device)

        real_distribution = self.ds.get_real_distribution()
        generated_distribution = self.g(_noise,
                                        self.ds[0][1].float().unsqueeze(0)
                                        .unsqueeze(2).to(self.device),  # dt
                                        self.ds[0][0].float().mean().to(self.device)) \
            .detach().cpu().view(1, -1).numpy()

        # Generate n samples with current form of the g
        for _stock, _dt in self.ds[1:]:
            _stock = _stock.float()
            _dt = _dt.float()

            generated_distribution = np.append(generated_distribution, self.g(_noise,
                                                                              _dt.unsqueeze(0)
                                                                              .unsqueeze(2).to(self.device),
                                                                              # dt
                                                                              _stock.mean().to(
                                                                                  self.device))  # mean
                                               .detach().cpu().view(1, -1).numpy(), axis=0)

        _generated_distribution = np.array(list(map(self.ds.mean_reshape, generated_distribution)))
        generated_distribution = _generated_distribution

        return real_distribution, generated_distribution

    def train_system(self):
        for epoch in range(self.num_epochs):
            for batch_idx, real in enumerate(self.dl):
                stock, dt = real

                stock = stock.view(*stock.shape, 1)
                stock = stock.float()
                stock = stock.to(self.device)

                mean = stock.mean()

                dt = dt.view(*dt.shape, 1)
                dt = dt.float()
                dt = dt.to(self.device)

                noise_stock = self.dist_latent.sample(sample_shape=(self.batch_size, self.seq_len, self.g_dim_latent))
                noise_stock = noise_stock.to(self.device)

                g_iter = self.g_iter
                d_iter = self.d_iter

                for var_name in range(g_iter):
                    loss_g, _ = self.train_generator(noise_stock, stock, dt, mean)

                for var_name in range(d_iter):
                    loss_d, fake = self.train_discriminator(noise_stock, stock, dt, mean)

                if batch_idx == 0:
                    print(
                        f"Epoch [{epoch}/{self.num_epochs}] Batch {batch_idx}/{len(self.dl)} \
                          Loss D: {loss_d.detach().cpu().item():.4f}, loss G: {loss_g.detach().cpu().item():.4f}"
                    )

                    # Visualize the whole distribution
                    rd, gd = self.generate_distributions()
                    fig = visualization_metrics.visualize(gd, rd)

                    wandb.log({
                        'epoch': epoch,
                        'd loss': loss_d.detach().cpu().item(),
                        'g loss': loss_g.detach().cpu().item(),
                        'mae': mean_absolute_error(stock[0].view(-1).cpu().numpy(),
                                                   fake[0].view(-1).detach().cpu().numpy()),
                        'Conditional on deltas fake sample': plot_time_series(
                            fake[0].view(-1).detach().cpu().numpy(),
                            '[Conditional (on deltas)] Fake sample'),
                        'Real sample': plot_time_series(
                            stock[0].view(-1).cpu().numpy(),
                            '[Corresponding] Real sample'),
                        'Distribution': fig
                    })


if __name__ == '__main__':
    # setup
    torch.random.manual_seed(42)

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # run_name = str('RCGAN: {} {} \n'.format(config['run_name'], datetime.datetime.now()) + str(config.values()))
    run_name = '1g / 16d | G mean condition'

    wandb.init(config=config, project='time-gan-2017', name=run_name)

    time_gan = TimeGAN(config)
    time_gan.train_system()

    # time_gan.g = time_gan.g.to('cpu')
    # time_gan.d = time_gan.d.to('cpu')
    #
    # torch.save(time_gan.g.state_dict(), './models/time_gan_2017_g.pt')
    # torch.save(time_gan.d.state_dict(), './models/time_gan_2017_d.pt')
    #
    # time_gan.g = time_gan.g.to(config['system']['device'])
    # time_gan.d = time_gan.d.to(config['system']['device'])
