import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import json
import torch


class NIPSDataSet(Dataset):
    MODELS = ['timegan', 'rcgan']

    def __init__(self, seq_len, data: np.ndarray, model: str):
        self.seq_len = seq_len
        self.raw_data = data
        self.model = model
        if self.model == 'timegan':
            pass
        elif self.model == 'rcgan':
            self.data = [torch.from_numpy(np.array(self.raw_data[i:i + self.seq_len]))
                         for i in range(0, len(self.raw_data) - self.seq_len)]

            # Compute ∆t (deltas)
            self.dt_data = [torch.from_numpy(np.concatenate([np.array([0]),
                                                             self.data[i][1:].numpy() -
                                                             self.data[i][:-1].numpy()]))
                            for i in range(len(self.data))]

            # Filter for small size chunks
            self.data = list(filter(lambda t: t.shape[0] == seq_len, self.data))
            self.dt_data = list(filter(lambda t: t.shape[0] == seq_len, self.dt_data))

            self.data = self.data[:len(self.data) - 2]  # size problem
            self.dt_data = self.dt_data[:len(self.dt_data) - 2]  # size problem
        else:
            exit(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.dt_data[item]


class GeneralDataset:
    PATHS = {
        'electricity': './electricity_nips/train/data.json',
        'solar': './solar_nips/train/train.json',
        'traffic': './traffic_nips/train/data.json',
        'exchange': './exchange_rate_nips/train/train.json',
        'taxi': './taxi_30min/train/train.json'
    }

    def __init__(self, seq_len: int, ds_type: str, model: str):
        self.seq_len = seq_len
        self.ds_type = ds_type
        self.model = model
        self.json_data = []
        self.path = self.PATHS[self.ds_type]

        with open(self.path) as f:
            for item in f:
                data = json.loads(item)
                self.json_data.append(data)

        self.data = pd.DataFrame(self.json_data)
        self.data = self.data.sort_values(by='start')
        self.timestamps = self.data['start']
        self.values = self.data['target']

    def get_dataset(self):
        return NIPSDataSet(seq_len=self.seq_len, data=self.values[0], model=self.model)


if __name__ == '__main__':
    ds_generator = GeneralDataset(150, 'electricity', 'rcgan')
    ds = ds_generator.get_dataset()
