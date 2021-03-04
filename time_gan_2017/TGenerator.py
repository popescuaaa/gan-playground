import torch
import torch.nn as nn

"""
    An LSTM based generator. It expects a sequence of noise vectors as input.
"""


class LSTMGenerator(nn.Module):

    def __init__(self, dim_latent, dim_output, num_layers=1, dim_hidden=256):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.dim_latent = dim_latent

        '''
         x = noise from dim latent
        lstm(
             dim_x,
             hidden state features dimension,
             how many lstm cells
             ) =>out and a tuple (hn, cn) => the nth output pairs (important order)
             
        out = tensor containing the output features (h_t) from the last layer of the LSTM
        '''
        self.lstm = nn.LSTM(self.dim_latent + 1, self.dim_hidden, self.num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(self.dim_hidden, self.dim_output))

        self.h_0 = nn.Parameter(torch.zeros(1, self.dim_hidden))
        self.c_0 = nn.Parameter(torch.zeros(1, self.dim_hidden))

    def forward(self, noise_stock, dt):
        batch_size, seq_len = noise_stock.size(0), noise_stock.size(1)

        z = torch.cat([noise_stock, dt], 2)

        h_0 = self.h_0.unsqueeze(0).repeat(batch_size, 1, 1).permute(1, 0, 2)
        c_0 = self.c_0.unsqueeze(0).repeat(batch_size, 1, 1).permute(1, 0, 2)

        recurrent_features, _ = self.lstm(z, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous())
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
