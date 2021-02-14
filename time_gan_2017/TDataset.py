from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
from random import choice  # for dt sampling

DATASET_CONFIG = ['Open' 'High' 'Low' 'Close' 'Adj_Close' 'Volume']


class StockDataset(Dataset):
    def __init__(self, seq_len=10, normalize=False):
        # sequence len is limited by numpy at 32

        self.row_data = np.loadtxt('./stock_data.csv', delimiter=',', dtype=np.str)

        self.data = [float(e[3]) for e in self.row_data[1:]]
        tensor_like = []

        for idx in range(0, len(self.data) - seq_len):
            tensor_like.append(self.data[idx: idx + seq_len])

        self.data = torch.FloatTensor(tensor_like)
        self.len = self.data.shape[0]
        self.mean = self.data.mean()
        self.std = torch.std(self.data)

        if normalize:
            self.data = self.normalize()

    def normalize(self):
        normalized_data = (self.data - self.mean) / self.std
        return normalized_data

    def denormalize(self):
        denormalized_data = self.data * self.std + self.mean
        return denormalized_data

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]


class StockDatasetDeltas(Dataset):
    """
    Dataset return a sequence: [ start_value : sequence_deltas ]
    Actual sequence len: seq_len - 1
    """

    def __init__(self, seq_len=10):
        # sequence len is limited by numpy at 32

        self.row_data = np.loadtxt('./stock_data.csv', delimiter=',', dtype=np.str)

        self.data = [float(e[3]) for e in self.row_data[1:]]
        tensor_like_data = []
        tensor_like_deltas = []

        for idx in range(0, len(self.data) - seq_len):
            deltas = np.subtract(np.array(self.data[idx: idx + seq_len][1:]),
                                 np.array(self.data[idx: idx + seq_len][:-1]))

            # Added starting value as 0 to match sizes in G and D
            deltas = np.concatenate([np.array([0]),
                                     deltas])

            tensor_like_deltas.append(deltas)
            tensor_like_data.append(self.data[idx: idx + seq_len])

        self.data = torch.FloatTensor(tensor_like_data)
        self.deltas = torch.FloatTensor(tensor_like_deltas)

        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx], self.deltas[idx]


def test_stock_dataset() -> None:
    ds = StockDataset()
    dl = DataLoader(ds, batch_size=5, shuffle=True, num_workers=10)
    for i, real in enumerate(dl):
        print('Index: {} data {}'.format(i, real))
        break

    ds = StockDataset(normalize=True)
    dl = DataLoader(ds, batch_size=10, shuffle=True, num_workers=10)
    for i, batch in enumerate(dl):
        print('Index: {} data {}'.format(i, batch))
        break


def test_deltas_stock_dataset() -> None:
    ds = StockDatasetDeltas()
    dl = DataLoader(ds, batch_size=5, shuffle=False, num_workers=10)
    for i, real in enumerate(dl):
        data, deltas = real
        print('Shapes: data -> {} | deltas: {}'.format(data.shape, deltas.shape))
        print('Index: {} data {}'.format(i, deltas))
        break


class SinWaveDataset(Dataset):
    def __init__(self, csv_path: str, seq_len: int):
        self.seq_len = seq_len
        self.df = pd.read_csv(csv_path)

        # Compute ∆t (deltas)
        self.dt = np.array([(self.df.Wave[i + 1] - self.df.Wave[i])
                            for i in range(self.df.Wave.size - 1)])
        self.dt = np.concatenate([np.array([0]), self.dt])

        # Append ∆t (deltas)
        self.df.insert(1, 'dt', self.dt)

        # Create two structures for data and ∆t
        self.sine_wave_data = [torch.from_numpy(np.array(self.df.Wave[i: i + self.seq_len]))
                               for i in range(self.df.size - self.seq_len)]

        self.dt_data = [torch.from_numpy(np.array(self.df.dt[i: i + self.seq_len]))
                        for i in range(self.df.size - self.seq_len)]

    def __len__(self):
        return len(self.sine_wave_data)

    def __getitem__(self, idx: int):
        return self.sine_wave_data[idx], self.dt_data[idx]

    def sample_dt(self):
        return choice(self.dt_data)
