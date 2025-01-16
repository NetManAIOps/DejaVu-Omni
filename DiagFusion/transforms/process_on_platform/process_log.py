r"""

源文件（包含log\service）：
  ,log_id,              ,timestamp      ,cmdb_id           ,log_name                            ,value
0  k3aBt38B4OJPLTSquRM4  1651334403      cartservice-0      log_cartservice-service_application  [40minfo: Microsoft.AspNet...

目标npy文件：
    [
        [
            ... log:list
            # 格式
            (timestamp(10), 'instance', 'template')
        ] // case 0
    ]
"""
# 配置
IGNORE_INSTANCE = ["redis-cart-0", "redis-cart-1", "redis-cart-2", "redis-cart2-0"]

# 代码
from sklearn.model_selection import train_test_split

import json
import logging
import sys
import time
import pandas as pd
import numpy as np
import random
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from tqdm import tqdm
from datetime import datetime


def hash_decimal_to_hex(decimal):
    # 使用Python内置哈希函数将整数哈希为一个大整数
    hashed_decimal = abs(hash(str(decimal)))
    # 将大整数转换为16进制字符串
    hex_string = hex(hashed_decimal)
    # 取字符串末尾8个字符作为哈希值，即一个长度为8的16进制数
    hash_value = hex_string[:8]
    # 将16进制数转换为整数并返回
    return hash_value


def get_drain_template(log_df):
    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    config = TemplateMinerConfig()
    config.load("drain3/drain.ini")
    config.profiling_enabled = True
    template_miner = TemplateMiner(config=config)

    lines = log_df["message"].tolist()

    result_json_list = []
    for line in tqdm(lines, desc="draining"):
        line = str(line).rstrip()

        result = template_miner.add_log_message(line)

        result_json_list.append(result)

    sorted_clusters = sorted(
        template_miner.drain.clusters, key=lambda it: it.size, reverse=True
    )
    for cluster in sorted_clusters:
        logger.info(cluster)

    EID_list = []
    for logdict in result_json_list:
        EID_list.append(hash_decimal_to_hex(logdict["cluster_id"]))

    log_df["EventId"] = EID_list

    return log_df


def stratified_sampling(df: pd.DataFrame, run_table, save_path):
    logs_list = []
    for i in tqdm(range(0, len(run_table)), desc="日志采样："):
        service_list = []
        temp_df = df.loc[
            (df["timestamp"] >= run_table["st_time"][i])
            & (df["timestamp"] <= run_table["ed_time"][i])
        ]
        unique_list = np.unique(temp_df["EventId"], return_counts=True)
        event_id = unique_list[0]
        cnt = unique_list[1]
        for k in range(len(cnt)):
            if cnt[k] == 1:
                unique_log = (
                    temp_df[temp_df["EventId"] == event_id[k]].T.to_dict().values()
                )
                unique_log = list(unique_log)[0]
                service_list.append(
                    [
                        unique_log["timestamp"],
                        unique_log["cmdb_id"],
                        unique_log["EventId"],
                    ]
                )
                temp_df = temp_df[temp_df["EventId"] != event_id[k]]
        X = temp_df
        y = temp_df["EventId"]

        class_num = len(event_id)

        if len(temp_df) == 0:
            logs_list.append([])
            continue
        elif len(temp_df) < class_num and len(temp_df) >= 1:
            X_test = temp_df
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=class_num, stratify=y
            )
        for _, row in X_test.iterrows():
            service_list.append([row["timestamp"], row["cmdb_id"], row["EventId"]])
        logs_list.append(service_list)
    np.save(save_path, logs_list)


def process_log(config, run_table):
    print("处理log")
    print("读取日志")
    import os

    log_df = None
    log_file_path_list = []
    for dirpath, _, filenames in os.walk(config["dataset_entry"]):
        for filename in filenames:
            if filename.find("log-") != -1:
                full_log_path = os.path.join(dirpath, filename)
                log_file_path_list.append(full_log_path)
    for path in log_file_path_list:
        log_data = pd.read_csv(path)
        if log_data is None:
            log_df = log_data
        else:
            log_df = pd.concat([log_df, log_data])
        print(f"成功处理{path}")
    log_df = log_df.query(f"cmdb_id not in {IGNORE_INSTANCE}")

    #  2023-11-06T16:25:43.833Z
    log_df["timestamp"] = log_df["timestamp"].apply(lambda x: int(x))
    # log_df["timestamp"] = log_df["timestamp"].apply(lambda x: int(datetime.strptime(x.split('.')[0], "%Y-%m-%dT%H:%M:%S").timestamp()))
    print("提取模板，准备drain")
    log_template_df = get_drain_template(log_df)
    print("准备采样")
    stratified_sampling(log_template_df, run_table, config["log_path"])
    print("处理完成")
