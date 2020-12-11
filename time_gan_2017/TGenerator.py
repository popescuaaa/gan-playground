import torch
import torch.nn as nn

"""
    An LSTM based generator. It expects a sequence of noise vectors as input.
"""


class LSTMGenerator(nn.Module):

    def __init__(self, dim_latent, dim_output, dist_latent, layers_num=1, dim_hidden=256):
        super().__init__()
        self.n_layers = layers_num
        self.hidden_dim = dim_hidden
        self.out_dim = dim_output
        self.dist_latent = dist_latent

        # x = noise from dim latent
        # lstm(
        #       dim_x,
        #      hidden state features dimension,
        #      how many lstm cells
        #      ) =>out and a tuple (hn, cn) => the nth output pairs (important order)
        # out = tensor containing the output features (h_t) from the last layer of the LSTM
        # we take the out value => linear and tanh => [-1, 1]

        self.lstm = nn.LSTM(dim_latent, dim_hidden, layers_num, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(dim_hidden, dim_output), nn.Tanh())

        # tanh => increase with a value form [-1, 1]

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # initial values

        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        recurrent_features, _ = self.lstm(x, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size * seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs

    def sample(self, num_samples) -> torch.Tensor:
        z = self.dist_latent.sample(sample_shape=(num_samples, self.dim_latent))
        return z

    def generate_visual_sample(self, num_samples):
        z = self.dist_latent.sample(num_samples)
        x = self.forward(z)
        return x