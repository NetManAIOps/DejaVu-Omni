import random
import fasttext
import numpy as np
import pandas as pd
import public_function as pf
from collections import Counter
import hashlib
import time
from tqdm import tqdm

class FastTextLab:
    def __init__(self, config, cases, split=True):
        self.config = config
        self.cases = cases
        if self.config["supervised"]:
            self.method = fasttext.train_supervised
        else:
            self.method = fasttext.train_unsupervised
        self.nodes = config["nodes"].split()
        self.anomaly_types = np.append("[normal]", cases["anomaly_type"].unique()).tolist()
        self.anomaly_types.sort()
        self.anomaly_type_labels = dict(
            zip(self.anomaly_types, range(len(self.anomaly_types)))
        )
        self.node_labels = dict(zip(self.nodes, range(len(self.nodes))))
        print(self.anomaly_type_labels)
        self.train_data, self.test_data = self.prepare_data()

    def prepare_data(self):
        metric_trace_text_path = self.config["text_path"]
        temp_data = pf.load(metric_trace_text_path)

        train = self.cases[self.cases["data_type"] == "train"].index
        test = self.cases[self.cases["data_type"] == "test"].index
        total = self.cases.index
        self.save_to_txt(temp_data, train, self.config["train_path"])
        self.save_to_txt(temp_data, test, self.config["test_path"])

        with open(self.config["train_path"], "r") as f:
            train_data = f.read().splitlines()
        with open(self.config["test_path"], "r") as f:
            test_data = f.read().splitlines()
        return train_data, test_data

    def w2v_DA(self):
        da_train_data = self.train_data.copy()
        model = self.method(
            self.config["train_path"],
            dim=self.config["vector_dim"],
            minCount=self.config["minCount"],
            minn=0,
            maxn=0,
            epoch=self.config["epoch"],
            seed=self.config["seed"],
            thread=1
        )
        

        sche = tqdm(total=len(self.anomaly_types) * len(self.nodes), desc='数据增强')

        for anomaly_type in self.anomaly_types:
            for node in self.nodes:
                sample_count = len(
                    [
                        text
                        for text in self.train_data
                        if text.split("__label__")[-1]
                        == str(self.node_labels[node])
                        + str(self.anomaly_type_labels[anomaly_type])
                    ]
                )
                if sample_count == 0:
                    sche.update(1)
                    continue
                anomaly_texts = [
                    text
                    for text in self.train_data
                    if text.split("\t")[-1]
                    == f"__label__{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}"
                ]
                loop = 0
                while sample_count < self.config["sample_count"]:
                    loop += 1
                    if loop >= 10 * self.config["sample_count"]:
                        break
                    # 随机选取相应label的序列进行复制
                    chosen_text, label = anomaly_texts[
                        random.randint(0, len(anomaly_texts) - 1)
                    ].split("\t")
                    chosen_text_splits = chosen_text.split()
                    if len(chosen_text_splits) < self.config["minCount"]:
                        continue
                    # 随机选取若干事件进行替换
                    edit_event_ids = random.sample(
                        range(len(chosen_text_splits)), self.config["edit_count"]
                    )
                    for event_id in edit_event_ids:
                        # 替换被选中的事件，选取离他距离最近的事件用于替换
                        nearest_event = model.get_nearest_neighbors(
                            chosen_text_splits[event_id]
                        )[0][-1]
                        chosen_text_splits[event_id] = nearest_event
                    da_train_data.append(
                        " ".join(chosen_text_splits)
                        + f"\t__label__{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}"
                    )
                    sample_count += 1
                
                sche.update(1)

        with open(self.config["train_da_path"], "w") as f:
            for text in da_train_data:
                f.write(text + "\n")

    def event_embedding_lab(self, data_path):
        if self.config["train"]:
            model = self.method(
                data_path,
                dim=self.config["vector_dim"],
                minCount=self.config["minCount"],
                minn=0,
                maxn=0,
                epoch=self.config["epoch"],
                seed=self.config["seed"],
                thread=1
            )
            model.save_model(self.config["model_save_path"])
        else:
            model = fasttext.load_model(self.config["model_save_path"])
        event_dict = dict()
        for event in model.words:
            event_dict[event] = model[event]
        return event_dict

    def save_to_txt(self, data: dict, keys, save_path):
        fillna = False
        with open(save_path, "w") as f:
            for case_id in keys:
                case_id = case_id if case_id in data.keys() else str(case_id)
                for node_info in data[case_id]:
                    text = data[case_id][node_info]
                    if isinstance(text, str):
                        text = text.replace("(", "").replace(")", "")
                        if fillna and len(text) == 0:
                            text = "None"
                        f.write(
                            f"{text}\t__label__{self.node_labels[node_info[0]]}{self.anomaly_type_labels[node_info[1]]}\n"
                        )
                    #                         self.anomaly_types.add(f'{node_info[0]}{node_info[1]}')

                    elif isinstance(text, list):
                        text = " ".join(text)
                        if fillna and len(text) == 0:
                            text = "None"
                        f.write(
                            f"{text}\t__label__{self.node_labels[node_info[0]]}{self.anomaly_type_labels[node_info[1]]}\n"
                        )
                    #                         self.anomaly_types.add(f'{node_info[0]}{node_info[1]}')

                    else:
                        raise Exception("type error")

        return

    def do_lab(self):
        if self.config["train"]:
            self.w2v_DA()
        pf.save(
            self.config["save_path"],
            self.event_embedding_lab(self.config["train_da_path"]),
        )


def run_fasttext(config, labels):
    # event embedding流程；基于数据增强
    print("Fasttext use seed:%d" % config["seed"])
    start_ts = time.time()
    lab2 = FastTextLab(config, labels)
    lab2.do_lab()
    end_ts = time.time()
    print("fasttext time used:", end_ts - start_ts, "s")
