"""
    Simple neural network for Iris classification
"""
from abc import ABC

import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

# graphs
import matplotlib as plt

# data loader
import numpy as np


class IrisDataset(Dataset):
    def __init__(self):
        self.raw_data = np.loadtxt('./Iris.csv', delimiter=',', dtype=np.str)
        self.len = self.raw_data.shape[0]
        self.data = [[int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), item[5]] \
                     for item in self.raw_data[1:len(self.raw_data)]]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class IrisNet(nn.Module):
    def __init__(self, config):
        super(IrisNet, self).__init__()

    def forward(self):
        pass



if __name__ == '__main__':
    dataset = IrisDataset()
    # load yaml data
    with open('simple_network.yaml', 'r') as f:
        config = yaml.load(f)
    
