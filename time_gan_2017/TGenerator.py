import torch
import torch.nn as nn

"""
    An LSTM based generator. It expects a sequence of noise vectors as input.
"""


class LSTMGenerator(nn.Module):

    def __init__(self, dim_latent: int, dim_output: int, num_layers: int = 1, dim_hidden: int = 256):
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
        self.linear = nn.Sequential(nn.Linear(self.dim_hidden + 1, self.dim_output, bias=True))

        self.h_0 = nn.Parameter(torch.zeros(1, self.dim_hidden))
        self.c_0 = nn.Parameter(torch.zeros(1, self.dim_hidden))

    def forward(self, noise, dt, mean):
        batch_size, seq_len = noise.size(0), noise.size(1)
        z = torch.cat([noise, dt], 2)
        h_0 = self.h_0.unsqueeze(0).repeat(batch_size, 1, 1).permute(1, 0, 2)
        c_0 = self.c_0.unsqueeze(0).repeat(batch_size, 1, 1).permute(1, 0, 2)
        recurrent_features, _ = self.lstm(z, (h_0, c_0))
        mean = mean.repeat(batch_size * seq_len).view(batch_size, seq_len, 1)
        _features = torch.cat([recurrent_features.contiguous(), mean], 2)
        outputs = self.linear(_features)
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


def run_generator_test():
    g = LSTMGenerator(128, 1)

    noise = torch.randn(size=(10, 150, 128))
    dt = torch.randn(size=(10, 150, 1))
    mean = torch.randn(size=(1,))

    generated_data = g(noise, dt, mean)

    print(generated_data.detach().numpy())


if __name__ == '__main__':
    run_generator_test()
