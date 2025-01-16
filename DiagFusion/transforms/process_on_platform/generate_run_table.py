r"""
处理groundtruth至run_table
(初赛)22AIops Challenge Dataset Anomaly Type:
pod-failure
memory
loss
delay
corrupt

文件范围：
    groundtruth文件夹下，包含groundtruth的文件
文件输入格式:
    (JSON) timestamp(10), service, cmdb_id, failure_type
文件输出格式:
    ( CSV) index, service, instance, anomaly_type([login failure]), st_time(timestamp-10), ed_time(timestamp-10), duration(600), data_type(train/test)

"""

# 相关文件配置

# CASE_ANOMALY_DICT = {
#     "[cpu]": 40,
#     "[io]": 50,
#     "[memory]": 40,
#     "[network]": 60,
#     "[process]": 27,
# }
GROUNDTRUTH_PATH = "platform_data/groundtruth.json"
TEST_RATE = .5

# TEST_RATE_DICT = {
#     "[cpu]": 0.2,
#     "[io]": 0.1,
#     "[memory]": 0.3,
#     "[network]": 0.1,
#     "[process]": 0.1,
# }

# 执行代码
def generate_run_table(config):
    print("开始生成run_table")
    import os
    import json
    import random
    import numpy as np
    import pandas as pd


    # 获取groundtruth文件列表
    file_list = os.listdir(config["raw_data"]["dataset_entry"] + "/groundtruth")
    groundtruth = None
    for file in file_list:
        with open(os.path.join(config["raw_data"]["dataset_entry"] + "/groundtruth", file), "r", encoding="utf8") as r:
            data = json.load(r)
        if groundtruth is None:
            groundtruth = data
        else:
            for key in groundtruth.keys():
                groundtruth[key].extend(data[key])

    run_table = pd.DataFrame(groundtruth)


    # 重命名
    run_table = run_table.rename(
        columns={
            "timestamp": "st_time",
            "failure_type": "anomaly_type",
            "cmdb_id": "instance",
        }
    )
    # 按时间排序
    run_table = run_table.sort_values(by="st_time")

    run_table["anomaly_type"] = run_table["anomaly_type"].apply(
        lambda x: "[" + x + "]"
    )


    run_table["instance"] = run_table["service"]
    run_table["ed_time"] = run_table["st_time"] + run_table["duration"]
    run_table.reset_index(drop=True, inplace=True)

    # 划分训练测试集
    run_table["data_type"] = ["train"] * len(run_table)
    anomaly_cnt = run_table.groupby(by="anomaly_type")["instance"].count()

    anomaly_cnt_dict = anomaly_cnt.to_dict()
    for anomaly, group in run_table.groupby(by="anomaly_type"):
        sample_cnt = int(anomaly_cnt_dict[anomaly] * TEST_RATE)
        group_index = group.index.to_list()
        test_choices = group_index[-sample_cnt:]
        for choice in test_choices:
            run_table.loc[choice, "data_type"] = "test"

    # run_table.drop(columns=["level"], inplace=True)

    run_table_dir = os.path.join(
        config["base_path"],
        config["demo_path"],
        config["label"],
    )
    run_table_path = os.path.join(
        config["base_path"],
        config["demo_path"],
        config["label"],
        config["he_dgl"]["run_table"],
    )
    if not os.path.exists(run_table_dir):
        os.makedirs(run_table_dir)
    run_table.to_csv(run_table_path)
    print("生成完毕")
