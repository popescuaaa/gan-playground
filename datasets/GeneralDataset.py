import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import json


class NIPSDataSet(Dataset):
    MODELS = ['timegan', 'rcgan']

    def __init__(self, seq_len, data: np.ndarray, model: str):
        self.seq_len = seq_len
        self.data = data
        self.model = model
        if self.model == 'timegan':
            pass
        elif self.model == 'rcgan':
            #  Compute conditional timestamps
            pass
        else:
            exit(-1)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class GeneralDataset():
    PATHS = {
        'electricity': './electricity_nips/train/data.json',
        'solar': './solar_nips/train/train.json',
        'traffic': './traffic_nips/train/data.json',
        'exchange': './exchange_rate_nips/train/train.json',
        'taxi': './taxi_30min/train/train.json'
    }

    def __init__(self, ds_type: str):
        self.ds_type = ds_type
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


if __name__ == '__main__':
    ds = GeneralDataset(150)
    for e in ds:
        t, v = e
        print(v)
        break
