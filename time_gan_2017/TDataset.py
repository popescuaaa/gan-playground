from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
from random import choice, shuffle  # for data sampling

DATASET_CONFIG = ['Open', 'High', 'Low', 'Close']  # for test purpose
STOCK_OFFSET = 9  # offset from default sequence length (dataset dependent)


# class SinWaveDataset(Dataset):
#     def __init__(self, csv_path: str, seq_len: int):
#         self.seq_len = seq_len
#         self.df = pd.read_csv(csv_path)
#
#         # Compute ∆t (deltas)
#         self.dt = np.array([(self.df.Wave[i + 1] - self.df.Wave[i])
#                             for i in range(self.df.Wave.size - 1)])
#
#         # Create two structures for data and ∆t
#         self.sine_wave_data = [torch.from_numpy(np.array(self.df.Wave[i: i + self.seq_len]))
#                                for i in range(self.df.shape[0] - self.seq_len - 1)]
#
#         self.dt_data = [torch.from_numpy(np.array(self.dt[i: i + self.seq_len - 1]))
#                         for i in range(self.df.shape[0] - self.seq_len - 1)]
#
#         # Filter for small size chunks
#         self.sine_wave_data = list(filter(lambda t: t.shape[0] == seq_len, self.sine_wave_data))
#         self.dt_data = list(filter(lambda t: t.shape[0] == seq_len, self.dt_data))
#
#         self.sine_wave_data = self.sine_wave_data[:len(self.sine_wave_data) - 2]  # size problem
#         self.dt_data = self.dt_data[:len(self.dt_data) - 2]  # size problem
#
#     def __len__(self):
#         return len(self.sine_wave_data)
#
#     def __getitem__(self, idx: int):
#         return self.dt_data
#
#     def sample_dt(self):
#         return choice(self.dt_data)


class StockDataset(Dataset):
    """
        The stock dataset is adapted to process any csv file that respects the stock market data format
        in terms of mandatory columns: Date (optional), Open, High, Low, Close, Adj_Close, Volume (optional)

    """

    def __init__(self, csv_path: str, seq_len: int, config: str):
        assert (config in DATASET_CONFIG), 'Config element is not supported'
        self.seq_len = seq_len
        self.df = pd.read_csv(csv_path)

        # Create two structures for data and ∆t
        self.stock_data = [torch.from_numpy(np.array(self.df[config][i:i + self.seq_len]))
                           for i in range(0, self.df.shape[0] - self.seq_len)]

        # Compute ∆t (deltas)
        self.dt_data = [torch.from_numpy(np.concatenate([np.array([0]),
                                                         self.stock_data[i][1:].numpy() -
                                                         self.stock_data[i][:-1].numpy()]))
                        for i in range(len(self.stock_data))]

        # Filter for small size chunks
        self.stock_data = list(filter(lambda t: t.shape[0] == seq_len, self.stock_data))
        self.dt_data = list(filter(lambda t: t.shape[0] == seq_len, self.dt_data))

        self.stock_data = self.stock_data[:len(self.stock_data) - STOCK_OFFSET]  # size problem
        self.dt_data = self.dt_data[:len(self.dt_data) - STOCK_OFFSET]  # size problem

        self.data = [(self.stock_data[i], self.dt_data[i]) for i in range(len(self.stock_data))]

    def __len__(self):
        return len(self.stock_data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def mean_reshape(self, arr: np.array):
        mean = np.mean(arr)
        return np.repeat(mean, self.seq_len)

    def get_real_distribution(self):
        real_distribution = np.array(list(map(lambda t: t.numpy(), self.stock_data)))
        return np.array(list(map(self.mean_reshape, real_distribution)))

    def sample(self):
        random_idx = choice([i for i in range(len(self.stock_data))])
        return self.stock_data[random_idx], self.dt_data[random_idx]


if __name__ == '__main__':
    path = './csv/AAPL.csv'
    ds = StockDataset(path, 150, 'Close')
    dl = DataLoader(ds, batch_size=10, shuffle=False)
    rd = ds.get_real_distribution()
    print('Real dist: {}'.format(len(rd)))
