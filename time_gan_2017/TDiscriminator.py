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

        self.lstm = nn.LSTM(self.dim_input, self.dim_hidden, self.num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(self.dim_hidden, 1))

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        h_0 = torch.zeros(self.num_layers, batch_size, self.dim_hidden).cuda()
        c_0 = torch.zeros(self.num_layers, batch_size, self.dim_hidden).cuda()

        # now retrieve the hidden states as the recurrent features from lstm
        # Basically, the output of a lstm also contains a pair (h_n, c_n) last hidden state, last cell state
        recurrent_features, _ = self.lstm(x, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size * seq_len, self.dim_hidden))
        outputs = outputs.view(batch_size, seq_len, 1)
        return outputs
