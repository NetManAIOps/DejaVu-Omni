r"""
处理groundtruth至run_table
------
(初赛)22AIops Challenge Dataset Anomaly Type:
    k8s容器网络延迟
    k8s容器写io负载
    k8s容器读io负载
    k8s容器cpu负载
    k8s容器网络资源包重复发送
    k8s容器进程中止
    k8s容器网络丢包
    k8s容器内存负载
    k8s容器网络资源包损坏
    node 磁盘写IO消耗
    node 内存消耗
    node节点CPU故障
    node 磁盘读IO消耗
    node节点CPU爬升
    node 磁盘空间消耗
------
要求(不考虑node、只考虑pod)：
------
CPU：
    k8s容器cpu负载
内存:
    k8s容器内存负载
IO:
    k8s容器写io负载
    k8s容器读io负载
进程:
    k8s容器进程中止
网络:
    k8s容器网络延迟
    k8s容器网络资源包重复发送
    k8s容器网络丢包
    k8s容器网络资源包损坏

文件范围：
    groundtruth文件夹下，包含groundtruth的文件
文件输入格式:
    (JSON) timestamp(10), level, cmdb_id, failure_type
文件输出格式:
    ( CSV) index, service, instance, anomaly_type([login failure]), st_time(timestamp-10), ed_time(timestamp-10), duration(600), data_type(train/test)
要求：
    cmdb_id -(cut 2-0 or -1?)-> service
    cmdb_id -> instance
    failure_type -> anomaly_type([login failure])
"""


# 相关文件配置
"""
-------
[cpu]         47    14
[io]          83    24
[memory]      46    13
[network]    235    70
[process]     27    8
"""

# ANOMALY_CASE_DICT = {
#     "[cpu]": 38,  # 14
#     "[io]": 40,  # 15
#     "[memory]": 37,  # 15
#     "[network]": 50,  #  15
#     "[process]": 22,  # 8
# }

# 187

INCLUDE_LEVEL = ["pod"]
ANOMALY_DICT = {
    "k8s容器网络延迟": "network",
    "k8s容器写io负载": "io",
    "k8s容器读io负载": "io",
    "k8s容器cpu负载": "cpu",
    "k8s容器网络资源包重复发送": "network",
    "k8s容器进程中止": "process",
    "k8s容器网络丢包": "network",
    "k8s容器内存负载": "memory",
    "k8s容器网络资源包损坏": "network",
}
TEST_RATE = .2
# TEST_CASES = {
#     "[cpu]": 9,  # .2
#     "[io]": 16,  # .2
#     "[memory]": 9,  # .2
#     "[network]": 15,  #  数量太多, 挑15
#     "[process]": 5,  # .2
# }

# 54


# 执行代码
def generate_run_table(config):
    print("开始生成run_table")
    import os
    import json

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

    orgin_len = len(groundtruth["timestamp"])
    # 将service数据扩为pod级别
    append_list = ["-0", "-1", "-2", "2-0"]
    for index, level in enumerate(groundtruth["level"]):
        if level == "service":
            timestamp = groundtruth["timestamp"][index]
            service = groundtruth["cmdb_id"][index]
            failure_type = groundtruth["failure_type"][index]
            for append in append_list:
                instance = service + append
                new_case = {
                    "timestamp": groundtruth["timestamp"][index],
                    "level": "pod",
                    "cmdb_id": instance,
                    "failure_type": failure_type,
                }
                for key in new_case.keys():
                    groundtruth[key].append(new_case[key])

    import pandas as pd

    run_table = pd.DataFrame(groundtruth)

    run_table = run_table.query(f"level in {INCLUDE_LEVEL}")

    for index in range(orgin_len, len(groundtruth["timestamp"])):
        run_table.loc[index, "level"] = "service"

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

    # 处理故障类型（合并）
    run_table["anomaly_type"] = run_table["anomaly_type"].apply(
        lambda x: "[" + ANOMALY_DICT[x] + "]"
    )

    run_table["duration"] = [540] * len(run_table)
    import re

    run_table["service"] = run_table["instance"].apply(
        lambda x: re.sub(r"\d?\-\d", "", x)
    )
    run_table["ed_time"] = run_table["st_time"] + run_table["duration"]
    run_table.reset_index(drop=True, inplace=True)

    # 划分训练测试集
    run_table["data_type"] = ["train"] * len(run_table)

    anomaly_cnt = run_table.groupby(by="anomaly_type")["instance"].count()
    # 固定选取每个 anomaly 的后 30% 的作为测试
    anomaly_cnt_dict = anomaly_cnt.to_dict()
    for anomaly, group in run_table.groupby(by="anomaly_type"):
        sample_cnt = int(anomaly_cnt_dict[anomaly] * TEST_RATE)
        group_index = group.index.to_list()
        test_choices = group_index[-sample_cnt:]
        for choice in test_choices:
            run_table.loc[choice, "data_type"] = "test"

    # # 固定选取每个 anomaly 的后 30% 的作为测试
    # for anomaly, group in run_table.groupby(by="anomaly_type"):
    #     sample_cnt = TEST_CASES[anomaly]
    #     group_index = group.index.to_list()
    #     test_choices = group_index[-sample_cnt:]
    #     for choice in test_choices:
    #         run_table.loc[choice, "data_type"] = "test"

    # test_cases = run_table[run_table["data_type"] == "test"]


    # train_cases = run_table[run_table["data_type"] == "train"]

    # new_train_cases = None
    # # 对训练集进行调整
    # for anomaly, group in train_cases.groupby(by="anomaly_type"):
    #     if len(group) > MAX_TRAIN_CASES:
    #         # cases 数量超过已定数值，则进行筛选
    #         group = group.sample(n=MAX_TRAIN_CASES)
    #     if new_train_cases is None:
    #         new_train_cases = group
    #     else:
    #         new_train_cases = pd.concat((new_train_cases, group))

    # run_table = pd.concat((new_train_cases, test_cases))
    run_table = run_table.sort_values(by="st_time")
    run_table = run_table.reset_index(drop=True)

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
