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

"""

index, 
service, 
instance, 
anomaly_type([login failure]), 
st_time(timestamp-10), 
ed_time(timestamp-10), 
duration(600), 
data_type(train/test)
"""

import pandas as pd
import re
from datetime import datetime

if __name__ == "__main__":
    run_table = pd.read_csv("./human_sample.csv", index_col=0)
    run_table = run_table.query("level in ['pod']")
    run_table = run_table.drop(columns=["timestamp", "time"])
    run_table = run_table.rename(
        columns={
            "cmdb_id": "instance",
            "failure_type": "anomaly_type",
            "start": "st_time",
            "end": "ed_time",
            "type": "data_type",
        }
    )
    run_table["st_time"] = run_table["st_time"].apply(
        lambda x: int(datetime.strptime(x, "%Y/%m/%d %H:%M").timestamp())
    )
    run_table["ed_time"] = run_table["ed_time"].apply(
        lambda x: int(datetime.strptime(x, "%Y/%m/%d %H:%M").timestamp())
    )

    run_table["duration"] = run_table["ed_time"] - run_table["st_time"]

    run_table["anomaly_type"] = run_table["anomaly_type"].apply(
        lambda x: ANOMALY_DICT[x]
    )
    run_table["service"] = run_table["instance"].apply(
        lambda x: re.sub(r"\d?\-\d", "", x)
    )

    # pod_cases = run_table.query("level == 'pod'")
    # service_cases = run_table.query("level == 'service'")
    # new_pod_cases = []
    # level,service,instance,anomaly_type,st_time,ed_time,data_type,duration
    # for _, case in service_cases.iterrows():
    #     append_list = ["-0", "-1", "-2", "2-0"]
    #     for append in append_list:
    #         instance = case["service"] + append
    #         new_pod_cases.append(
    #             {
    #                 "service": case["service"],
    #                 "instance": instance,
    #                 "level": case["level"],
    #                 "anomaly_type": case["anomaly_type"],
    #                 "st_time": case["st_time"],
    #                 "ed_time": case["ed_time"],
    #                 "data_type": case["data_type"],
    #                 "duration": case["duration"],
    #             }
    #         )
    # run_table = pd.concat((pod_cases, pd.DataFrame(new_pod_cases)))
    run_table = run_table.reset_index(drop=True)

    run_table.to_csv("./run_table.csv")
