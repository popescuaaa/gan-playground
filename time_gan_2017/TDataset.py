from torch.utils.data import Dataset
import numpy as np

DATASET_CONFIG = ['Open' 'High' 'Low' 'Close' 'Adj_Close' 'Volume']


class StockDataset(Dataset):
    def __init__(self):
        self.row_data = np.loadtxt('./stock_data.csv', delimiter=',', dtype=np.str)
        self.len = self.row_data.shape[0] - 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.row_data[idx]
