import torch.nn as nn
import torch


# noinspection DuplicatedCode
class LSTMDiscriminator(nn.Module):
    """An LSTM based discriminator. It expects a sequence as input and outputs a probability for each element.
    """

    def __init__(self, dim_input, num_layers=1, dim_hidden=256):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.dim_input = dim_input

        self.lstm = nn.LSTM(self.dim_input + 1, self.dim_hidden, self.num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(self.dim_hidden, 1))

        self.h_0 = nn.Parameter(torch.zeros(1, self.dim_hidden))
        self.c_0 = nn.Parameter(torch.zeros(1, self.dim_hidden))

    def forward(self, stock, dt, mean, std):
        batch_size, seq_len = stock.size(0), stock.size(1)

        _stock = torch.cat([stock, dt], 2)
        stock = _stock

        h_0 = self.h_0.unsqueeze(0).repeat(batch_size, 1, 1).permute(1, 0, 2)
        c_0 = self.c_0.unsqueeze(0).repeat(batch_size, 1, 1).permute(1, 0, 2)

        # now retrieve the hidden states as the recurrent features from lstm
        # Basically, the output of a lstm also contains a pair (h_n, c_n) last hidden state, last cell state
        recurrent_features, _ = self.lstm(stock, (h_0, c_0))
        recurrent_features = recurrent_features[:, -1, :]

        # mean = mean.repeat(batch_size).view(batch_size, 1)
        # std = std.repeat(batch_size).view(batch_size, 1)

        #features = torch.cat([recurrent_features.contiguous(), mean, std], 1)
        outputs = self.linear(recurrent_features)
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
