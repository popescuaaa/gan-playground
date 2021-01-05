from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

DATASET_CONFIG = ['Open' 'High' 'Low' 'Close' 'Adj_Close' 'Volume']


class StockDataset(Dataset):
    def __init__(self, normalize=False):
        self.row_data = np.loadtxt('./stock_data.csv', delimiter=',', dtype=np.str)
        self.data = [float(e[3]) for e in self.row_data[1:]]
        self.len = len(self.data)
        self.max = max(self.data)

        if normalize:
            self.data = self.normalize()

    def normalize(self):
        normalized_data = [float(e / self.max) for e in self.data]
        return normalized_data

    def denormalize(self):
        denormalized_data = [float(e * self.max) for e in self.data]
        return denormalized_data

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    # dataset test
    ds = StockDataset()
    dl = DataLoader(ds, batch_size=10, shuffle=True, num_workers=10)
    for i, batch in enumerate(dl):
        print('Index: {} data {}'.format(i, batch))
        break

    ds = StockDataset(normalize=True)
    dl = DataLoader(ds, batch_size=10, shuffle=True, num_workers=10)
    for i, batch in enumerate(dl):
        print('Index: {} data {}'.format(i, batch))
        break
