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
        self.lstm = nn.LSTM(self.dim_latent, self.dim_hidden, self.num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(self.dim_hidden, self.dim_output))

        # from linear <-
        self.h_0 = nn.Parameter(torch.zeros(1, self.dim_hidden))
        self.c_0 = nn.Parameter(torch.zeros(1, self.dim_hidden))

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        h_0 = self.h_0.unsqueeze(0).repeat(batch_size, 1, 1).permute(1, 0, 2)
        c_0 = self.c_0.unsqueeze(0).repeat(batch_size, 1, 1).permute(1, 0, 2)

        recurrent_features, _ = self.lstm(x, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous())
        return outputs
