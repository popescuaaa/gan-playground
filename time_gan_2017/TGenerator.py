import torch
import torch.nn as nn

"""
    An LSTM based generator. It expects a sequence of noise vectors as input.
"""


class LSTMGenerator(nn.Module):

    def __init__(self, dim_latent, dim_output, dist_latent, num_layers=1, dim_hidden=256):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.dist_latent = dist_latent
        self.dim_latent = dim_latent

        # x = noise from dim latent
        # lstm(
        #       dim_x,
        #      hidden state features dimension,
        #      how many lstm cells
        #      ) =>out and a tuple (hn, cn) => the nth output pairs (important order)
        # out = tensor containing the output features (h_t) from the last layer of the LSTM
        # we take the out value => linear and tanh => [-1, 1]

        self.lstm = nn.LSTM(dim_latent, self.dim_hidden,  self.num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(self.dim_hidden, self.dim_output), nn.Tanh())

        # tanh => increase with a value form [-1, 1]

    def forward(self, x):
        print('arrive_here')
        batch_size, seq_len = x.size(0), x.size(1)

        # initial values

        h_0 = torch.zeros(self.num_layers, batch_size, self.dim_hidden)
        c_0 = torch.zeros(self.num_layers, batch_size, self.dim_hidden)

        recurrent_features, _ = self.lstm(x, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size * seq_len, self.num_layers))
        outputs = outputs.view(batch_size, seq_len, self.dim_output)
        return outputs

    def sample(self, num_samples) -> torch.Tensor:
        z = self.dist_latent.sample(sample_shape=(num_samples, self.dim_latent))
        return z

    def generate_visual_sample(self, num_samples):
        z = self.dist_latent.sample(num_samples)
        x = self.forward(z)
        return x