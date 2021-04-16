import json
import pandas as pd

DATASETS = [
    'traffic_nips',
    'taxi_30min',
    'exchange_rate_nips',
    'electricity_nips',
    'solar_nips']


if __name__ == '__main__':
    with open('./solar_nips/train/train.json') as f:
        json_data = []
        for line in f:
            data = json.loads(line)
            json_data.append(data)

        df = pd.DataFrame(json_data)
        sorted_df = df.sort_values(by='start')
        print(sorted_df.values[0])
