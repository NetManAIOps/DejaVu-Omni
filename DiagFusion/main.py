from transforms.events import (
    fasttext_with_DA,
    sententce_embedding,
    metric_trace_log_parse,
)
from models import He_DGL
from public_function import deal_config, get_config
import os
import pandas as pd
import torch
import numpy as np
import random
from datetime import datetime

def set_seed(config):
    seed = config["fasttext"]["seed"]
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


if __name__ == "__main__":
    start = datetime.now()
    config = get_config()

    set_seed(config)

    label_path = os.path.join(
        config["base_path"],
        config["demo_path"],
        config["label"],
        config["he_dgl"]["run_table"],
    )
    labels = pd.read_csv(label_path, index_col=0)

    print("[parse]")
    metric_trace_log_parse.run_parse(deal_config(config, "parse"), labels)

    print("[fasttext]")
    fasttext_with_DA.run_fasttext(deal_config(config, "fasttext"), labels)

    print("[sentence_embedding]")
    sententce_embedding.run_sentence_embedding(
        deal_config(config, "sentence_embedding")
    )

    print("[dgl]")
    lab_id = 9  # 实验唯一编号
    he_dgl = He_DGL.UnircaLab(deal_config(config, "he_dgl"))
    if config["he_dgl"]["train"]:
        he_dgl.do_lab(lab_id)
    else:
        he_dgl.test(lab_id)

    end = datetime.now()
    print("cost :%.2fs" % (end.timestamp() - start.timestamp()))