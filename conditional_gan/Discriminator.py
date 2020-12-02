import torch.nn as nn
import torch

# TODO: add configuration for norm and activation

'''
    dim_input: input dimension / size
    dim_output: output dimension / size
'''


class Discriminator(nn.Module):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_labels = 10

        # Condition
        self.label_emb = nn.Embedding(self.num_labels, self.num_labels)

        __module_list = [
            nn.Linear(self.dim_input + self.num_labels, self.dim_input // 2, bias=True),
            nn.BatchNorm1d(self.dim_input // 2, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(self.dim_input // 2, self.dim_input // 4, bias=True),
            nn.BatchNorm1d(self.dim_input // 4, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(self.dim_input // 4, self.dim_input // 8, bias=True),
            nn.BatchNorm1d(self.dim_input // 8, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(self.dim_input // 8, self.dim_output, bias=True)
        ]

        self.__net = nn.Sequential(*__module_list)

    def forward(self, x, labels):
        # Condition
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)  # stack tensors
        out = self.__net(x)
        return out.squeeze()
