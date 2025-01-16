import json
import math
import public_function as pf
import pandas as pd
import numpy as np
from tqdm import tqdm


def metric_trace_log_parse(trace, metric, logs, labels, save_path, nodes):
    if not metric is None: # 去除np.inf数值的指标
        for k, v in metric.items():
            metric[k] = [x for x in v if not math.isinf(x[3])]

    if not logs is None:
        logs = list(logs)
        log = {x: [] for x in labels.index}
        if labels.index[-1]+1 == len(log):
            for k, v in log.items():
                log[k] = logs[int(k)]
        else:
            count = 0
            for k, v in log.items():
                log[k] = logs[count]
                count += 1

    # service_name = sorted(list(set(labels['service'])))
#     service_name = np.load('/home/u2120210568/jupyterfiles/zhangbicheng/unirca/data/21aiops/nodes.pkl', allow_pickle=True) # 仅针对21数据集
    service_name = nodes.split()
    anomaly_service = list(labels['instance'])
    anomaly_type = list(labels['anomaly_type'])

#     demo_metric = {x: {} for x in metric.keys()}
    demo_metric = {x: {} for x in labels.index}
    k = 0
    for case_id, v in tqdm(demo_metric.items()):
        anomaly_service_name = anomaly_service[k]
        anomaly_service_type = anomaly_type[k]
        k += 1
        inner_dict_key = [(x, anomaly_service_type) if x == anomaly_service_name else (x, "[normal]") for x in
                          service_name]
        # 指标
        if not metric is None:
            demo_metric[case_id] = {x: [[y[0], "{}_{}_{}".format(y[1], y[2], "+" if y[3] > 0 else "-")] for y in metric[str(case_id)] if
                                  y[1].find(x[0]) != -1] for x in inner_dict_key}
        else:
            demo_metric[case_id] = {x : [] for x in inner_dict_key}
        # 调用链
        if not trace is None:
            for inner_key in inner_dict_key:
                demo_metric[case_id][inner_key].extend(
                    [[y[0], "{}_{}".format(y[1], y[2])] for y in trace[str(case_id)] 
                     if y[1] == inner_key[0] or y[2] == inner_key[0]])
        # 日志
        if not logs is None:
            for inner_key in inner_dict_key:
                demo_metric[case_id][inner_key].extend([[y[0], y[2]] for y in log[case_id] if y[1] == inner_key[0]])
        for inner_key in inner_dict_key:
            temp = demo_metric[case_id][inner_key]
            sort_list = sorted(temp, key=lambda x: x[0])
            temp_list = [x[1] for x in sort_list]
            demo_metric[case_id][inner_key] = ' '.join(temp_list)

    pf.save(save_path, demo_metric)


def run_parse(config, labels):
    trace = None
    metric = None
    logs = None
    if config['log_path']:
        logs = np.load(config['log_path'], allow_pickle=True)
    if config['metric_path']:
        with open(config['metric_path'], 'r', encoding='utf8') as fp:
            metric = json.load(fp)
    if config['trace_path']:
        with open(config['trace_path'], 'r', encoding='utf8') as fp:
            trace = json.load(fp)
    metric_trace_log_parse(trace, metric, logs, labels, config['save_path'], config['nodes'])
