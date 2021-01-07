from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np

DATASET_CONFIG = ['Open' 'High' 'Low' 'Close' 'Adj_Close' 'Volume']


class StockDataset(Dataset):
    def __init__(self, seq_len=10, normalize=False):
        # sequence len is limited by numpy at 32

        self.row_data = np.loadtxt('./stock_data.csv', delimiter=',', dtype=np.str)

        self.data = [float(e[3]) for e in self.row_data[1:]]
        tensor_like = []

        for idx in range(0, len(self.data) // seq_len, seq_len):
            tensor_like.append(self.data[idx: idx + seq_len])

        self.data = torch.FloatTensor(tensor_like)

        self.len = self.data.shape[1]
        self.max = self.data.max()

        # There is the same asset price but they can be grouped in small chunks
        # of length seq_len as they will irl by date

        if normalize:
            self.data = self.normalize()

    def normalize(self):
        # change for [0, 1]
        normalized_data = self.data / self.max
        return normalized_data

    def denormalize(self):
        denormalized_data = [float(e * self.max) for e in self.data]
        return denormalized_data

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]


# if __name__ == '__main__':
#     # dataset test
#     ds = StockDataset()
#     dl = DataLoader(ds, batch_size=10, shuffle=True, num_workers=10)
#     for i, real in enumerate(dl):
#         print('Index: {} data {}'.format(i, real))
#         print(real.shape)
#         break
#
#     ds = StockDataset(normalize=True)
#     dl = DataLoader(ds, batch_size=10, shuffle=True, num_workers=10)
#     for i, batch in enumerate(dl):
#         print('Index: {} data {}'.format(i, batch))
#         break
