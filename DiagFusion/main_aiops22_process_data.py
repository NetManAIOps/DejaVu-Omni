r"""
该文件是用于处理22年挑战赛初赛数据的可执行main文件
"""
from transforms.process_on_aiops import (
    generate_run_table,
    process_log,
    process_metric,
    process_trace,
)
from public_function import get_config, deal_config
import pandas as pd
import torch
import numpy as np
import random

def set_seed(config):
    seed = config['fasttext']['seed']
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)

if __name__ == "__main__":
    import os

    config = get_config()
    set_seed(config)

    run_table_path = os.path.join(
        config["base_path"],
        config["demo_path"],
        config["label"],
        config["he_dgl"]["run_table"],
    )

    generate_run_table.generate_run_table(config)

    run_table = pd.read_csv(run_table_path)

    # 创建文件夹
    store_dir = os.path.join(
        config["base_path"],
        config["demo_path"],
        config["label"],
        config["raw_data"]["store_dir"],
    )
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    # 处理log
    process_log.process_log(deal_config(config, "raw_data"), run_table)
    # 处理metric
    process_metric.process_metric(deal_config(config, "raw_data"), run_table)
    # 处理trace
    process_trace.process_trace(deal_config(config, "raw_data"), run_table_path)
