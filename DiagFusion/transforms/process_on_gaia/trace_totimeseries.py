import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..')))
import pandas as pd
import numpy as np
import time
from datetime import datetime
import pytz
import json
from detector.k_sigma import Ksigma
import public_function as pf
from tqdm import tqdm

tz = pytz.timezone('Asia/Shanghai')


def ts_to_date(timestamp):
    try:
        return datetime.fromtimestamp(timestamp//1000, tz).strftime('%Y-%m-%d %H:%M:%S.%f')
    except:
        return datetime.fromtimestamp(timestamp//1000, tz).strftime('%Y-%m-%d %H:%M:%S')


def time_to_ts(ctime):
    try:
        timeArray = time.strptime(ctime, '%Y-%m-%d %H:%M:%S.%f')
    except:
        try:
            timeArray = time.strptime(ctime, '%Y-%m-%d %H:%M:%S')
        except:
            timeArray = time.strptime(ctime, '%Y-%m-%d')
    return int(time.mktime(timeArray)) * 1000


class TraceUtils:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pairs = self.getPairs()
        
    def getPairs(self):
        services_pairs = {'webservice': ['mobservice', 'redisservice'], 'mobservice': ['redisservice'], 
                          'logservice': ['dbservice', 'redisservice'], 'dbservice': ['redisservice']} 
        pairs = []
        for caller in services_pairs:
            for callee in services_pairs[caller]:
                for i in [1, 2]:
                    for j in [1, 2]:
                        pairs.append((caller+str(i), callee+str(j)))
        pairs.extend([('logservice1', 'logservice2'), ('logservice2', 'logservice1')])
        return pairs
    
    def get_trace_by_day(self, day: int):
        day = f'0{day}' if day < 10 else str(day)
        temp = []
        for service in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, 
                                                 f'{service}/trace_{service}_2021-07-{day}.csv')
            if not os.path.exists(filepath):
                continue
            temp.append(pd.read_csv(filepath, index_col=0))
        return pd.concat(temp, ignore_index=True)

    # lagency需要减去子调用的最大值，但是暂时没有减去，因此耗时太久
    def data_process(self, day: int):
        data = self.get_trace_by_day(day)
        # lagency暂时没有减去
    #     cdata = data.groupby(['trace_id', 'parent_id'], as_index=False).max()
        cdata = data[[
            'parent_id', 'service_name'
        ]].rename(columns={'parent_id': 'span_id', 'service_name': 'cservice_name'})
        return pd.merge(data, cdata, on='span_id')
    
    # 将lagency按照调用队整理成时间序列
    def turn_to_timeseries(self, day, savepath):
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        df = self.data_process(day)
        df['timestamp'] = df['timestamp'].apply(time_to_ts)
        date = ts_to_date(df['timestamp'][0]).split()[0]
        start_ts = time_to_ts(date)
        delta = 30000 # 30s per point
        points_count = 60000*24*60 // delta
        ts = [start_ts+delta*i for i in range(1, points_count+1)]
        for caller, callee in self.pairs:
            print(caller, callee)
            temp = df.loc[(df['service_name']==caller)&
                          (df['cservice_name']==callee)]
            info = {'timestamp': ts, '200': [], '500': [], 'other': [], 'lagency': []}
            for k in range(points_count):
                chosen = temp.loc[(temp['timestamp']>=start_ts+k*delta)&
                                (temp['timestamp']<start_ts+(k+1)*delta)]
                cur_lagency = max(0, np.mean(chosen['lagency'].values))
                cur_200 = len(chosen.loc[chosen['status_code']==200])
                cur_500 = len(chosen.loc[chosen['status_code']==500])
                cur_other = len(chosen) - cur_200 - cur_500
                info['lagency'].append(cur_lagency)
                info['200'].append(cur_200)
                info['500'].append(cur_500)
                info['other'].append(cur_other)
            pd.DataFrame(info).to_csv(os.path.join(savepath, f'{caller}_{callee}.csv'), index=False)   

class InvocationEvent:
    def __init__(self, cases, data_dir, dataset, trace_pairs_path, config=None):
        self.cases = cases
        self.data_dir = data_dir
        self.dataset = dataset
        self.trace_pairs_path = trace_pairs_path
        self.detector = Ksigma()
        self.pairs = self.getPairs()
        if config is None:
            config = {}
            config['minute'] = 60000
            config['MIN_TEST_LENGTH'] = 5
        self.config = config
        self.res = self.res = dict(zip(list(cases.index), [[] for _ in range(len(cases))]))
        
    def getPairs(self):
        if self.dataset == 'gaia':
            microServices = []
            for service in ['webservice', 'mobservice', 'logservice', 'dbservice', 'redisservice']:
                microServices.extend([service+str(1), service+str(2)])

            # services_pairs = {'webservice': ['mobservice', 'redisservice'], 'mobservice': ['redisservice'],
            #                   'logservice': ['dbservice', 'rediservice'], 'dbservice': ['redisservice'],
            #                  'logservice': ['redisservice']}
            services_pairs = {'webservice': ['mobservice', 'redisservice'], 'mobservice': ['redisservice'],
                            'logservice': ['dbservice', 'redisservice'], 'dbservice': ['redisservice']}
            pairs = []
            for caller in services_pairs:
                for callee in services_pairs[caller]:
                    for i in [1, 2]:
                        for j in [1, 2]:
                            pairs.append((caller+str(i), callee+str(j)))
            pairs.extend([('logservice1', 'logservice2'), ('logservice2', 'logservice1')])
        elif self.dataset == '20aiops':
            with open(self.trace_pairs_path, 'r') as f:
                pairs= [eval(line.rstrip('\n')) for line in f]
        else:
            raise Exception()
        return pairs
    
    def read(self, day, caller, callee):
        filepath = os.path.join(self.data_dir, str(day), f'{caller}_{callee}.csv')
        data = pd.read_csv(filepath)
        data.index = [ts_to_date(ts) for ts in data['timestamp']]
        return data
    
    def get_invocation_events(self):
        # latency异常和状态码500的数目异常
        if self.dataset == 'gaia':
            for case_id, case in tqdm(self.cases.iterrows()):
                day = int(case['datetime'].split('-')[-1])
                for caller, callee in self.pairs:
                    invocation_data = self.read(day, caller, callee)
                    # 故障前一分钟至故障结束后一分钟
                    start_ts = time_to_ts(case['st_time'])-self.config['minute']*31
                    end_ts = time_to_ts(case['ed_time'])+self.config['minute']*1
                    res1 = self.detector.detection(invocation_data, 'lagency', start_ts, end_ts)
                    res2 = self.detector.detection(invocation_data, '500', start_ts, end_ts)
    #                 print(res1, res2)
                    if not (res1[0] or res2[0]):
                        continue
                    ts = None
                    if res1[0]:
                        ts = res1[1]
                        score = res1[2]
                    if res2[0]:
                        if ts is None:
                            ts = res2[1]
                        else:
                            ts = min(ts, res2[1])
                        if ts == res2[1]:
                            score = res2[2] 
                    self.res[case_id].append((int(ts), caller, callee, score))
        elif self.dataset == '20aiops':
            for case_id, case in tqdm(self.cases.iterrows()):
                day = case['st_time'].split(" ")[0]
                for caller, callee in self.pairs:
                    temp_csv = os.path.join(self.data_dir, day, f'{caller}_{callee}.csv')
                    if not os.path.exists(temp_csv):
                        continue
                    invocation_data = self.read(day, caller, callee)
                    # 故障前一分钟至故障结束后一分钟
                    start_ts = time_to_ts(case['st_time'])-self.config['minute']*31
                    end_ts = time_to_ts(case['ed_time'])+self.config['minute']*1
                    res1 = self.detector.detection(invocation_data, 'lagency', start_ts, end_ts)
                    res2 = self.detector.detection(invocation_data, 'other', start_ts, end_ts)
    #                 print(res1, res2)
                    if not (res1[0] or res2[0]):
                        continue
                    ts = None
                    if res1[0]:
                        ts = res1[1]
                        score = res1[2]
                    if res2[0]:
                        if ts is None:
                            ts = res2[1]
                        else:
                            ts = min(ts, res2[1])
                        if ts == res2[1]:
                            score = res2[2] 
                    self.res[case_id].append((int(ts), caller, callee, score))
        else:
            raise Exception()
                
    def save_res(self, savepath):
        with open(savepath, 'w') as f:
            json.dump(self.res, f)
        print('Save successfully!')

# demo样例
"""
demopath = '/home/jinpengxiang/jupyterfiles/xiasibo/GAIA/data/lab/demo.csv'
data_dir = 'trace_temp/trace_data'
demo_labels = pd.read_csv(demopath)
invocation_event = InvocationEvent(demo_labels, data_dir)
invocation_event.get_invocation_events()
invocation_event.save_res('demo_trace_only_status.json')
"""

if __name__ == '__main__':
    config = pf.get_config()
    project_root_dir = os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0],
                                                    '../..'))
    label_path = os.path.abspath(os.path.join(project_root_dir, 
                                              config['base_path'], 
                                              config['demo_path'],
                                              config['label'], 'demo.csv'))
    trace_data_dir = config['trace_data_dir']
    labels = pd.read_csv(label_path)
    invocation_event = InvocationEvent(labels, trace_data_dir, config['dataset'], config['trace_pairs_path'])
    invocation_event.get_invocation_events()
    
    save_path = os.path.abspath(os.path.join(project_root_dir, 
                            pf.deal_config(config, 'parse')['trace_path']))
    invocation_event.save_res(save_path)