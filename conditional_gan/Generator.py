import torch.nn as nn
import torch

# TODO: add configurability for norm and activation function
# TODO: device configuration for work on cluster
'''
    dist_latent: latent distribution for sampling and testing
    
    cfg: config file containing
        dim_latent: latent space dimension / size
        dim_hidden: hidden layer preset dimension / size
        output_size: preset output size
'''


class Generator(nn.Module):
    def __init__(self, dist_latent, dim_latent, dim_hidden, dim_output):
        super(Generator, self).__init__()

        self.dist_latent = dist_latent
        self.dim_latent = dim_latent
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.num_labels = 10

        __module_list = [
            nn.Linear(self.dim_latent + self.num_labels, self.dim_hidden * 16, bias=True),
            nn.BatchNorm1d(self.dim_hidden * 16, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(self.dim_hidden * 16, self.dim_hidden * 32, bias=True),
            nn.BatchNorm1d(self.dim_hidden * 32, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(self.dim_hidden * 32, self.dim_hidden * 64, bias=True),
            nn.BatchNorm1d(self.dim_hidden * 64, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(self.dim_hidden * 64, self.dim_output, bias=True),
            nn.Tanh()
        ]

        self.__net = nn.Sequential(*__module_list)

    def forward(self, z, c):
        _z = torch.cat([z, c], 1)
        out = self.__net(_z)
        return out

    def sample(self, num_samples) -> torch.Tensor:
        z = self.dist_latent.sample(sample_shape=(num_samples, self.dim_latent))
        return z

    def generate_visual_sample(self, num_samples, c):
        z = self.sample(num_samples).cuda()
        x = self.forward(z, c)
        return x
