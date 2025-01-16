from collections import defaultdict
from functools import partial
from typing import Set, List, Any, Optional
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np


def top_k_accuracy(y_true: List[Set[Any]], y_pred: List[List[Any]], k=1, printer=None):
    assert len(y_true) == len(y_pred)
    cnt = 0
    for a, b in zip(y_true, y_pred):
        left = a
        right = set(b[:k])
        if len(left) != 0 and left <= right:
            cnt += 1
        else:
            if printer:
                printer(f"expected: {left}, actual: {right}")
    return cnt / max(len(y_true), 1)


top_1_accuracy = partial(top_k_accuracy, k=1)
top_2_accuracy = partial(top_k_accuracy, k=2)
top_3_accuracy = partial(top_k_accuracy, k=3)


def get_rank(y_true: Set[Any], y_pred: List[Any], max_rank: Optional[int] = None) -> List[float]:
    if len(y_true) == 0:
        return [ max_rank+1 ]
    rank_dict = defaultdict(lambda: len(y_pred) + 1 if max_rank is None else (max_rank + len(y_pred)) / 2)
    for idx, item in enumerate(y_pred, start=1):
        if item in y_true:
            rank_dict[item] = idx
    return [rank_dict[_] for _ in y_true]

def get_reverse_rank(y_true: Set[Any], y_pred: List[Any], max_rank: Optional[int] = None) -> List[float]:
    if len(y_true) == 0:
        return [ 1.0 / (max_rank+1) ]
    rank_dict = defaultdict(lambda: 1.0 / (len(y_pred) + 1) if max_rank is None else 2.0 / (max_rank + len(y_pred)))
    for idx, item in enumerate(y_pred, start=1):
        if item in y_true:
            rank_dict[item] = 1.0 / idx
    return [rank_dict[_] for _ in y_true]


# noinspection PyPep8Naming
def MFR(y_true: List[Set[Any]], y_pred: List[List[Any]], max_rank: Optional[int] = None):
    return np.mean([
        np.min(get_rank(a, b, max_rank))
        for a, b in zip(y_true, y_pred)
    ])


# noinspection PyPep8Naming
def MAR(y_true: List[Set[Any]], y_pred: List[List[Any]], max_rank: Optional[int] = None):
    return np.mean([
        np.mean(get_rank(a, b, max_rank))
        for a, b in zip(y_true, y_pred)
    ])

def MRR(y_true: List[Set[Any]], y_pred: List[List[Any]], max_rank: Optional[int] = None):
    return np.mean([
        np.mean(get_reverse_rank(a, b, max_rank))
        for a, b in zip(y_true, y_pred)
    ])

def Precision_Recall_F1(y_true: List[Set[Any]], y_pred: List[List[Any]], average = "weighted"):
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(y_true)
    y_pred = mlb.transform([[p] for p in y_pred])
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return precision, recall, f1



# def get_evaluation_metrics_dict(y_true: List[Set[Any]], y_pred: List[List[Any]], max_rank: Optional[int] = None):
#     metrics = {
#         "A@1": top_1_accuracy(y_true, y_pred),
#         "A@2": top_2_accuracy(y_true, y_pred),
#         "A@3": top_3_accuracy(y_true, y_pred),
#         "A@5": top_k_accuracy(y_true, y_pred, k=5),
#         "MAR": MAR(y_true, y_pred, max_rank=max_rank),
#         "MFR": MFR(y_true, y_pred, max_rank=max_rank),
#     }
#     return metrics

def get_evaluation_metrics_dict(y_trues: List[Set[Any]], y_preds: List[List[Any]], cdp, prefix=None):
    print(f"all {cdp.n_components} components")
    rcl_label_list: List[List] = [set([cdp.gid_to_component_id(_) for _ in labels]) for labels in y_trues]
    rcl_pred_list: List[List] = [[cdp.gid_to_component_id(_) for _ in preds] for preds in y_preds]
    rcl_pred_dedepulicated_list = []
    for preds in rcl_pred_list:
        seen = set()
        tmp = []
        for item in preds:
            if item not in seen:
                tmp.append(item)
                seen.add(item)
        rcl_pred_dedepulicated_list.append(tmp)
    fc_label_list: List[List] = [set([cdp.gid_to_fc_id(_) for _ in labels]) for labels in y_trues]
    fc_pred_list: List[int] = [cdp.gid_to_fc_id(preds[0]) for preds in y_preds]
    fc_p_r_f = Precision_Recall_F1(fc_label_list, fc_pred_list)
    metrics = {
        "A@1": top_1_accuracy(y_trues, y_preds),
        "A@2": top_2_accuracy(y_trues, y_preds),
        "A@3": top_3_accuracy(y_trues, y_preds),
        "A@5": top_k_accuracy(y_trues, y_preds, k=5),
        "MAR": MAR(y_trues, y_preds, max_rank=cdp.n_failure_instances),
        "RCL_A@1": top_1_accuracy(rcl_label_list, rcl_pred_dedepulicated_list),
        "RCL_A@2": top_2_accuracy(rcl_label_list, rcl_pred_dedepulicated_list),
        "RCL_A@3": top_3_accuracy(rcl_label_list, rcl_pred_dedepulicated_list),
        "RCL_A@5": top_k_accuracy(rcl_label_list, rcl_pred_dedepulicated_list, k=5),
        "RCL_MAR": MAR(rcl_label_list, rcl_pred_dedepulicated_list, max_rank=cdp.n_components),
        "RCL_MRR": MRR(rcl_label_list, rcl_pred_dedepulicated_list, max_rank=cdp.n_components),
        "FC_Precision": fc_p_r_f[0],
        "FC_Recall": fc_p_r_f[1],
        "FC_F1": fc_p_r_f[2]
    }
    if prefix is not None:
        new_metrics = {f'{prefix}_{key}': value for key, value in metrics.items()}
        return new_metrics
    return metrics


rca_evaluation_metrics = {
    "A@1": top_1_accuracy,
    "A@2": top_2_accuracy,
    "A@3": top_3_accuracy,
    "MAR": MAR,
    "MFR": MFR,
}

__all__ = [
    'rca_evaluation_metrics',
    "top_1_accuracy",
    "top_2_accuracy",
    "top_3_accuracy",
    "top_k_accuracy",
    "MAR", "MFR", "MRR", "Precision_Recall_F1",
    "get_evaluation_metrics_dict",
]
