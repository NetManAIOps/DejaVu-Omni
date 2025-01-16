import pandas as pd
import numpy as np
from tqdm import tqdm

class Ksigma:
    def __init__(self, config=None):
        if config is None:
            config = {}
#             config['minute'] = 60000
#             config['his_len'] = config['minute']*60*2
#             config['MIN_TEST_LENGTH'] = 5
            # k sigma
            config['k_s'] = {}
            config['k_s']['k_thr'] = 3
            config['k_s']['std_thr'] = 0.1
            config['k_s']['win_size'] = 60 # 120
        self.config = config
    
    def detection(self, data: pd.DataFrame, column: str, start_ts: int, end_ts: int):
        """
        return
            is_anomalous[True or False], anomaly_start_ts[ts], anomaly_score
        """
        config = self.config
        test_data = data.loc[(data['timestamp']>=start_ts)&(data['timestamp']<end_ts)]
        win_size, k_thr, std_thr = config['k_s']['win_size'], config['k_s']['k_thr'], config['k_s']['std_thr']
        del config
        if len(test_data) < win_size:
            return False, None, 0
        if column not in data.columns:
            raise ValueError('column not in data.')
        values = test_data[column].values
        timestamps = test_data['timestamp'].values
        win_values = list(values[:win_size])
        mean = np.mean(win_values)
        std = max(np.std(win_values), std_thr*mean)
        for i in range(win_size, len(values)):
            if abs(values[i]-mean) > k_thr*std:
                if k_thr*std != 0:
                    anomaly_score = (values[i]-mean) / (k_thr*std)
                else:
                    anomaly_score = float('inf')
                return True, timestamps[i], anomaly_score
            win_values.append(values[i])
            del win_values[0]
            mean = np.mean(win_values)
            std = max(np.std(win_values), std_thr*mean)
        return False, None, 0
