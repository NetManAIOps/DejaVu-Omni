r"""
输入trace文件
,     timestamp,    cmdb_id,         span_id,                        trace_id,duration,type,status_code,operation_name,parent_span
1,1651507200154,frontend2-0,8f0740b095193dab,a9b794341cd885f4c3064cdf53f2c6e4,6961,rpc,0,hipstershop.ProductCatalogService/GetProduct,d9e27cef838d4587
输出：
{
    0: [
        [timestamp(10), caller, callee, value]
    ]
}
'adservice-0 adservice-1 adservice-2 adservice2-0 cartservice-0 
cartservice-1 cartservice-2 cartservice2-0 checkoutservice-0 checkoutservice-1
checkoutservice-2 checkoutservice2-0 currencyservice-0 currencyservice-1 currencyservice-2 
currencyservice2-0 emailservice-0 emailservice-1 emailservice-2 emailservice2-0
frontend-0 frontend-1 frontend-2 frontend2-0 paymentservice-0
paymentservice-1 paymentservice-2 paymentservice2-0 productcatalogservice-0 productcatalogservice-1
productcatalogservice-2 productcatalogservice2-0 recommendationservice-0 recommendationservice-1 recommendationservice-2
recommendationservice2-0 shippingservice-0 shippingservice-1 shippingservice-2 shippingservice2-0'

adservice
cartservice
checkoutservice
currencyservice
emailservice
frontend
paymentservice
productcatalogservice
recommendationservice
shippingservice

"""
# 代码
from copy import copy
from detector.k_sigma import Ksigma
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os
import multiprocessing


def sub_task(run_table_path, trace_file_path, index):
    try:
        run_table = pd.read_csv(run_table_path)
        trace_df = pd.read_csv(trace_file_path)
        detector = Ksigma()
        # 转换时间戳
        trace_df["timestamp"] = trace_df["timestamp"].apply(lambda x: int(x / 1000))

        trace_df = trace_df.rename(columns={"parent_span": "parent_id"})
        # 父子拼接
        meta_df = trace_df[["parent_id", "cmdb_id"]].rename(
            columns={"parent_id": "span_id", "cmdb_id": "ccmdb_id"}
        )
        trace_df = pd.merge(trace_df, meta_df, on="span_id")

        # 按事件排序
        trace_df = trace_df.sort_values(by="timestamp")
        time_series = trace_df["timestamp"].values.tolist()

        st_time = copy(time_series[0])
        ed_time = copy(time_series[-1])

        sche = tqdm(total=len(trace_df), desc=f"调用链收集{trace_file_path}")

        # 每个 60s | 1min 统计一次
        interval = 60
        trace_dict = {}
        for case_id, case in run_table.iterrows():
            trace_dict[case_id] = []

        sample_cnt = int((ed_time - st_time) / interval) + 1
        interval_info = {
            "caller": [],
            "callee": [],
            "timestamp": [],
            "ec":[],
            "lagency": [],
        }

        for caller, caller_group in trace_df.groupby(by="cmdb_id"):
            for callee, callee_group in caller_group.groupby(by="ccmdb_id"):
                for stop_point in range(sample_cnt):
                    sample_time = st_time + stop_point * interval
                    chosen = callee_group[
                        (callee_group["timestamp"] >= sample_time)
                        & (callee_group["timestamp"] < sample_time + interval)
                    ]
                    if len(chosen) == 0:
                        continue
                    cur_lagency = max(0, np.mean(chosen["duration"].values.tolist()))
                    cur_ec = len(chosen.query("status_code not in ['0','200','Ok','OK',0,200]"))
                    interval_info["caller"].append(caller)
                    interval_info["callee"].append(callee)
                    interval_info["ec"].append(cur_ec)
                    interval_info["lagency"].append(cur_lagency)
                    interval_info["timestamp"].append(sample_time)
                sche.update(len(callee_group))
        sche = tqdm(total=len(run_table), desc=f"调用链处理{trace_file_path}")
        interval_info = pd.DataFrame(interval_info)
        for case_id, case in run_table.iterrows():
            # 故障前60分钟至故障结束后0分钟
            cst_time = case["st_time"] - 60 * 60
            ced_time = case["ed_time"] + 60 * 0
            case_df = interval_info[
                (interval_info["timestamp"] >= cst_time)
                & (interval_info["timestamp"] < ced_time)
            ]
            for caller, caller_group in case_df.groupby(by="caller"):
                for callee, callee_group in caller_group.groupby(by="callee"):
                    res1 = detector.detection(
                        callee_group, "lagency", cst_time, ced_time
                    )
                    res2 = detector.detection(callee_group, "ec", cst_time, ced_time)
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
                    trace_dict[case_id].append((int(ts), caller, callee, score))
            sche.update(1)
        return trace_dict
    except Exception as e:
        print(e)


def process_trace(config, run_table_path):
    print("处理trace")

    # test_trace = "aiops22/2022-05-09/cloudbed/trace/all/trace_jaeger-span.csv"
    trace_file_path_list = [
        # "aiops22/2022-05-07/cloudbed/trace/all/trace_jaeger-span.csv",
        # "aiops22/2022-05-09/cloudbed/trace/all/trace_jaeger-span.csv",
    ]
    for dirpath, _, filenames in os.walk(config["dataset_entry"]):
        for filename in filenames:
            if filename.find("trace_jaeger-span") != -1:
                trace_file_path_list.append(os.path.join(dirpath, filename))

    pool = multiprocessing.Pool(processes=len(trace_file_path_list))

    tks = []
    for index, filepath in enumerate(trace_file_path_list):
        tk = pool.apply_async(sub_task, (run_table_path, filepath, index))
        tks.append(tk)

    pool.close()
    pool.join()
    trace_dict = None
    for tk in tks:
        data = tk.get()
        if trace_dict is None:
            trace_dict = data
        else:
            for key in trace_dict.keys():
                trace_dict[key].extend(data[key])

    with open(config["trace_path"], "w") as w:
        json.dump(trace_dict, w)
    print("处理完成")
