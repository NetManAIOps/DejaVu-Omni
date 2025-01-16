import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..')))
import json
import pandas as pd
from detector.k_sigma import Ksigma
from datetime import datetime
import pytz
import time
from tqdm import tqdm
import public_function as pf

tz = pytz.timezone('Asia/Shanghai')
def ts_to_date(timestamp):
    try:
        return datetime.fromtimestamp(timestamp, tz).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return datetime.fromtimestamp(timestamp//1000, tz).strftime('%Y-%m-%d %H:%M:%S')

def time_to_ts(ctime):
    try:
        timeArray = time.strptime(ctime, '%Y-%m-%d %H:%M:%S.%f')
    except:
        timeArray = time.strptime(ctime, '%Y-%m-%d')
    return int(time.mktime(timeArray))*1000

class MetricEvent:
    def __init__(self, cases, metric_path, data_dir, dataset='gaia', config=None):
        self.cases = cases
        self.periods = ['2021-07-01_2021-07-15', '2021-07-15_2021-07-31']
        if dataset == 'gaia':
            self.metrics = self.get_all_metric_names(pd.read_csv(metric_path))
        elif dataset == '20aiops': 
            self.metrics = os.listdir(data_dir)
        else:
            raise Exception(f'Unknown dataset {dataset}')
        self.data_dir = data_dir
        self.detector = Ksigma()
        self.dataset = dataset
        if config is None:
            config = {}
            config['minute'] = 60000
            config['MIN_TEST_LENGTH'] = 5
        self.config = config
#         self.savepath = savepath
        self.res = dict(zip(list(cases.index), [[] for _ in range(len(cases))]))
    
    def read(self, metric):
        data = pd.read_csv(os.path.join(self.data_dir, metric))
        data.index = [ts_to_date(ts) for ts in data['timestamp']]
        return data
    
    def get_all_metric_names(self, metrics_info):
        metric_names = []
        for index, row in metrics_info.iterrows():
            for period in self.periods:
                metric_names.append('_'.join([row['name'], period+'.csv']))
        return metric_names
    
    def get_metric_events(self):
        # 减少文件读取次数
        for metric in tqdm(self.metrics):
            metric_data = self.read(metric)
            case_ids = 0
            for case_id, case in self.cases.iterrows():
                if self.dataset == 'gaia':
                    start_ts = time_to_ts(case['st_time'])-self.config['minute']*40
                    end_ts = time_to_ts(case['ed_time'])+self.config['minute']*10
                elif self.dataset == '20aiops':
                    # 故障开始前取65个点，故障结束后取两个点
                    interval = int(metric.split('-')[-1].replace('.csv', ''))
                    before_min = interval * 65
                    after_min = interval * 2
                    start_ts = time_to_ts(case['st_time'])-self.config['minute']*before_min
                    end_ts = time_to_ts(case['ed_time'])+self.config['minute']*after_min
                else:
                    raise Exception(f'Unknown dataset {self.dataset}')
                res = self.detector.detection(metric_data, 'value', start_ts, end_ts)
                if res[0] is True:
                    metric_splits = metric.split('_')
                    if self.dataset == 'gaia':
                        name = '_'.join(metric_splits[2: -2])
                        service = metric_splits[0]
                        address = metric_splits[1]
                    elif self.dataset == '20aiops':
                        name = metric_splits[2].strip('.csv')
                        service = metric_splits[1]
                        address = metric_splits[0]
                    else:
                        raise Exception(f'Unknown dataset {self.dataset}')
                    self.res[case_id].append((int(res[1]), f'{address}_{service}', name, res[2]))
                    
    def save_res(self, savepath):
        with open(savepath, 'w') as f:
            json.dump(self.res, f)
        print('Save successfully!')


"""
示例:
# DATASET = 'gaia'
# demopath = '/home/jinpengxiang/jupyterfiles/zhangbicheng/unirca/data/gaia/run_table.csv'
# data_dir =   '/home/jinpengxiang/data/GAIA-DataSet/MicroSS/metric/'
# metric_path = '/home/jinpengxiang/jupyterfiles/xiasibo/GAIA/data/metric_screening/metric.csv'
DATASET = '20aiops'
demopath = '/home/jinpengxiang/jupyterfiles/zhangbicheng/unirca/data/20aiops/demo.csv'
data_dir =  '/home/jinpengxiang/jupyterfiles/zhangbicheng/unirca/data/20aiops/metrics/'
metric_path = None  # 20aiops指标不进行选择因此没有metric_path
demo_labels = pd.read_csv(demopath)
metric_event = MetricEvent(demo_labels, metric_path, data_dir, DATASET)
metric_event.get_metric_events()
metric_event.save_res('./20aiops_metric.json')
"""

if __name__ == '__main__':
    config = pf.get_config()
    project_root_dir = os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0],
                                                    '../..'))
    label_path = os.path.abspath(os.path.join(project_root_dir, 
                                              config['base_path'], 
                                              config['demo_path'],
                                              config['label'], 'demo.csv'))
    labels = pd.read_csv(label_path)
    metric_info_path = config['metric_info_path']
    metric_data_dir = config['metric_data_dir']
    dataset = config['dataset']
    # 获取metric_event
    metric_event = MetricEvent(labels, metric_info_path, metric_data_dir, dataset)
    metric_event.get_metric_events()
    save_path = os.path.abspath(os.path.join(project_root_dir, 
                               pf.deal_config(config, 'parse')['metric_path']))
    metric_event.save_res(save_path)
    