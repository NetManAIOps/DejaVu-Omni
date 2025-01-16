r"""
输入：
    container， 包含（kpi）
    timestamp,cmdb_id,kpi_name,value
1651420740,node-6.shippingservice2-0,container_cpu_cfs_periods,17.0
输出：（JSON）
    {
        0: [
            [timestamp(10), cmdb_id, kpi_name, value],
            []
        ]
    }

"""
# 配置
IGNORE_INSTANCE = ["redis-cart-0", "redis-cart-1", "redis-cart-2", "redis-cart2-0"]

# 代码
from tqdm import tqdm
from detector.k_sigma import Ksigma
import pandas as pd
import json
import multiprocessing


def process_task(df, case_id, st_time, ed_time):
    detector = Ksigma()
    rt = []
    scheduler = tqdm(total=len(df), desc=f"case:{case_id}, detecting")
    for instance, ins_group in df.groupby(by="cmdb_id"):
        if instance in IGNORE_INSTANCE:
            # 忽略掉没用的instance
            scheduler.update(len(ins_group))
            continue
        for kpi, kpi_group in ins_group.groupby(by="kpi_name"):
            res = detector.detection(kpi_group, "value", st_time, ed_time)
            if res[0] is True:
                rt.append([int(res[1]), instance, kpi, res[2]])
        scheduler.update(len(ins_group))
    return rt


def process_metric(config, run_table: pd.DataFrame):
    print("处理metric")
    import os

    metric_file_path_list = []
    for dirpath, _, filenames in os.walk(config["dataset_entry"]):
        for filename in filenames:
            if filename.find("kpi_container_") != -1:
                metric_file_path_list.append(os.path.join(dirpath, filename))
    metric_df = None
    for filepath in tqdm(metric_file_path_list, desc="加载metric数据"):
        data = pd.read_csv(filepath)
        filename = filepath.split("/")[-1]
        kpi_name = "_".join(filename.split("_")[1:]).rstrip(".csv")
        data["kpi_name"] = len(data) * [kpi_name]
        data["cmdb_id"] = data["cmdb_id"].apply(lambda x: x.split(".")[1])
        if metric_df is None:
            metric_df = data
        else:
            metric_df = pd.concat([metric_df, data])

    metric_dict = {}
    tasks = []
    pool = multiprocessing.Pool(processes=10)
    for case_id, case in run_table.iterrows():
        # 故障前60个点，故障后0个点
        sample_interval = 60
        st_time = case["st_time"] - (sample_interval * 60)
        ed_time = case["ed_time"] + (sample_interval * 0)
        task = pool.apply_async(
            process_task,
            (
                metric_df.query(f"timestamp >= {st_time} & timestamp < {ed_time}"),
                case_id,
                st_time,
                ed_time,
            ),
        )
        tasks.append((case_id, task))
        # 每个实例，每个指标采样
    pool.close()
    pool.join()
    for case_id, task in tasks:
        metric_dict[case_id] = task.get()

    with open(config["metric_path"], "w") as w:
        json.dump(metric_dict, w)

    print("处理完成")
