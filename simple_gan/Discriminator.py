import torch.nn as nn

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

        __module_list = [
            nn.Linear(self.dim_input, self.dim_input // 2, bias=True),
            # nn.BatchNorm1d(self.dim_input // 2, affine=True, track_running_stats=True),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.dim_input // 2, self.dim_input // 4, bias=True),
            # nn.BatchNorm1d(self.dim_input // 4, affine=True, track_running_stats=True),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.dim_input // 4, self.dim_input // 8, bias=True),
            # nn.BatchNorm1d(self.dim_input // 8, affine=True, track_running_stats=True),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.dim_input // 8, self.dim_output, bias=True)
        ]

        self.__net = nn.Sequential(*__module_list)

    def forward(self, x):
        return self.__net(x)
