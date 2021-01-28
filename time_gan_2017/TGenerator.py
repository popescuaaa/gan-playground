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

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        h_0 = torch.zeros(self.num_layers, batch_size, self.dim_hidden).cuda()
        c_0 = torch.zeros(self.num_layers, batch_size, self.dim_hidden).cuda()
        recurrent_features, _ = self.lstm(x, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size * seq_len, self.dim_hidden))
        outputs = outputs.view(batch_size, seq_len, self.dim_output)
        return outputs
