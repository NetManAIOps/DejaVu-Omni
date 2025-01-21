import json
import os
import numpy as np
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from loguru import logger
from typing import List, DefaultDict
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
import torch.nn.functional as func
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from DejaVu.evaluation_metrics import get_rank
from DejaVu.models.interface import DejaVuModelInterface
from failure_dependency_graph import FDG


def plot_scatter(clusters, color_pic, plot_rc_types, x_pic, cluster_cs_pic, cluster_rs_pic, save_path):
    cluster_cs_pic = [item.tolist() for item in cluster_cs_pic]
    with open(os.path.join(os.path.dirname(save_path), 'plot_dict.json'), 'w') as f:
        plot_dict = {
            'clusters': clusters,
            'color_pic': color_pic,
            'plot_rc_types': plot_rc_types,
            'x_pic': x_pic.tolist(),
            'cluster_cs_pic': cluster_cs_pic,
            'cluster_rs_pic': cluster_rs_pic,
            'save_dir': os.path.dirname(save_path),
        }
        json.dump(plot_dict, f)
    plt.figure(figsize=(10, 8))
    color_pic = np.array(color_pic)
    color_index = 10
    rc_type_color = {}
    rc_type_color['non-recur'] = 'red'
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    cluster_list = list(clusters.keys())
    for cluster in cluster_list:
        value = clusters[cluster]
        for i in value:
            color_pic[i] = color_index
        color_index += 1
    for i in range(10, color_index):
        cluster = cluster_list[i-10]
        plt.scatter(x_pic[np.where(color_pic==i), 0], x_pic[np.where(color_pic==i), 1], marker='o', color=colors[i-10], label=cluster, s=20)
        rc_type_color[cluster] = colors[i-10]
        c = cluster_cs_pic[i-10]
        r = cluster_rs_pic[i-10]
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = c[0] + r * np.cos(theta)
        y = c[1] + r * np.sin(theta)
        plt.plot(x, y, color=colors[i-10])
    # markersize = 25
    # for rc_type, color in rc_type_color.items():
    #     if rc_type != 'non-recur':
    #         indexes_true, indexes_false = [], []
    #         for i, temp in enumerate(plot_rc_types):
    #             if temp==rc_type:
    #                 if color_pic[i] == 1:
    #                     indexes_true.append(i)
    #                 elif color_pic[i] == 3:
    #                     indexes_false.append(i)
    #         plt.scatter(x_pic[indexes_true, 0], x_pic[indexes_true, 1], marker='*', color=color, label=rc_type, s=markersize)
    #         plt.scatter(x_pic[indexes_false, 0], x_pic[indexes_false, 1], marker='x', color=color, label=rc_type, s=markersize)
    #     else:
    #         plt.scatter(x_pic[np.where(color_pic==4), 0], x_pic[np.where(color_pic==4), 1], marker='*', color=color, label='non-recurring', s=markersize)
    #         plt.scatter(x_pic[np.where(color_pic==5), 0], x_pic[np.where(color_pic==5), 1], marker='x', color=color, label='non-recurring', s=markersize)
    # plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    # categories = ['true recurring', 'true non-recurring', 'false non-recurring', 'false recurring']
    legend_elements = [mlines.Line2D([], [], color=colors[i], marker='o', linestyle='None', 
                                   markersize=10, label=cluster_list[i]) for i in range(len(cluster_list))]
    # legend_elements.append(mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='non-recurring'))
    plt.legend(handles=legend_elements, frameon=True, shadow=True, fontsize=11, ncol=len(cluster_list)+1, loc='lower center', bbox_to_anchor=(0.5, 1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def get_recur_fscore(df: pd.DataFrame, threshold, output_path, beta=1.):
    x, x2 = [], []
    y, y2 = [], []
    non_recur = df[df.failure_type=='non-recur']
    non_recur_prob1 = list(non_recur['prob-1'])
    x.extend(non_recur_prob1)
    y.extend([1 for _ in range(len(non_recur_prob1))])
    recur = df[df.failure_type=='recur']
    recur_prob1 = list(recur['prob-1'])
    x.extend(recur_prob1)
    y.extend([0 for _ in range(len(recur_prob1))])
    x = np.array(x)
    y = np.array(y)
    y_pred = (x < threshold)
    precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred, beta=beta, average='binary')

    x2.extend(non_recur_prob1)
    y2.extend([1 for _ in range(len(non_recur_prob1))])
    recur_correct = df[(df.failure_type=='recur') & (df.correct=='True')]
    recur_correct_prob1 = list(recur_correct['prob-1'])
    x2.extend(recur_correct_prob1)
    y2.extend([0 for _ in range(len(recur_correct_prob1))])
    x2 = np.array(x2)
    y2 = np.array(y2)
    y2_pred = (x2 < threshold)
    precision2, recall2, fscore2, support2 = precision_recall_fscore_support(y2, y2_pred, beta=beta, average='binary')
    
    result_log = logger.add(output_path)
    logger.info(
        f"include recurring and wrong-predicted ones\nfscore: {fscore}, precision: {precision}, recall: {recall}\n"
        f"exclude recurring and wrong-predicted ones\nfscore: {fscore2}, precision: {precision2}, recall: {recall2}\n"
    )
    logger.remove(result_log)


def get_recur_threshold(df: pd.DataFrame, output_path, beta=1.):
    x = []
    y = []
    non_recur = df[df.failure_type=='non-recur']
    non_recur_prob1 = list(non_recur['prob-1'])
    x.extend(non_recur_prob1)
    y.extend([1 for _ in range(len(non_recur_prob1))])
    recur_correct = df[(df.failure_type=='recur') & (df.correct=='True')]
    recur_correct_prob1 = list(recur_correct['prob-1'])
    x.extend(recur_correct_prob1)
    y.extend([0 for _ in range(len(recur_correct_prob1))])
    x = np.array(x)
    y = np.array(y)
    best_fscore, best_precision, best_recall, threshold = 0, 0, 1, 0
    for value in x:
        y_pred = (x < value)
        precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred, beta=beta, average='binary')
        if best_fscore < fscore:
            best_fscore, best_precision, best_recall = fscore, precision, recall
            threshold = value
    right_fscore, right_precision, right_recall, right = best_fscore, best_precision, best_recall, threshold
    best_fscore, best_precision, best_recall, threshold = 0, 0, 1, 0
    for value in x:
        y_pred = (x < value+1e-6)
        precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred, beta=beta, average='binary')
        if best_fscore < fscore:
            best_fscore, best_precision, best_recall = fscore, precision, recall
            threshold = value+1e-6
    left_fscore, left_precision, left_recall, left = best_fscore, best_precision, best_recall, threshold
    threshold = 0
    if right_fscore > left_fscore:
        threshold = right
    elif right_fscore < left_fscore:
        threshold = left
    else:
        threshold = (left+right)/2
    result_log = logger.add(output_path)
    logger.info(
        f"threshold: [{left}, {right}]\n"
        f"right_fscore: {right_fscore}, right_precision: {right_precision}, right_recall: {right_recall}\n"
        f"left_fscore: {left_fscore}, left_precision: {left_precision}, left_recall: {left_recall}\n"
        f"threshod select: {threshold}\n"
    )
    logger.remove(result_log)
    return threshold


def detect_nonrecur_mhgl(model: DejaVuModelInterface, output_dir: Path, beta=1.):
    output_dir = output_dir
    output_dir.mkdir(exist_ok=True)
    cdp: FDG = model.train_dataset.cdp
    failure_dataloader = model.train_dataloader()
    feats_list, labels_list = [], []
    for batch_data in failure_dataloader:
        features, labels, failure_ids, graphs = batch_data
        feats: th.Tensor = model.module.get_agg_feat(features, graphs)
        valid_idx = th.where(th.any(labels, dim=1))[0]
        feats_list.append(feats.detach()[valid_idx])
        labels_list.append(labels.detach()[valid_idx])
    feats_cluster = th.concat(feats_list)
    logger.debug(f"{feats_cluster.shape=}")
    labels_list = [set(th.where(label >= 1)[0].tolist()) for label in th.concat(labels_list)]
    clusters = DefaultDict(list)
    i = 0
    feats_cluster_ext = []
    for feats, labels in zip(feats_cluster, labels_list):
        # one instance can be delivered to multiple clusters.
        for label in labels:
            rc_type = cdp.instance_to_class(cdp.gid_to_instance(label))
            feats_cluster_ext.append(feats[None,label,:])
            clusters[rc_type].append(i)
            i += 1
    feats_cluster = th.concat(feats_cluster_ext)
    centers = {}
    rads = {}
    for cluster, value in clusters.items():
        center = th.mean(feats_cluster[value], dim=0)
        dists = th.norm(feats_cluster[value]-center, p=2, dim=-1)
        rad = th.max(dists)
        centers[cluster] = center
        rads[cluster] = rad.item()

    pca = PCA(n_components=2)
    pca.fit(feats_cluster)
    feats_reduced = pca.transform(feats_cluster)
    color_pic = [0 for _ in range(feats_reduced.shape[0])]
    plot_rc_types = ['' for _ in range(feats_reduced.shape[0])]
    cluster_cs_pic = []
    cluster_rs_pic = []
    for cluster, value in clusters.items():
        tmp = feats_reduced[value, :]
        cluster_c_pic = np.mean(tmp, axis=0)
        tmp_d = np.linalg.norm(tmp-cluster_c_pic, ord=2, axis=-1)
        rad = np.max(tmp_d)
        cluster_cs_pic.append(cluster_c_pic)
        cluster_rs_pic.append(rad)

    non_recurring_list = model.non_recurring_list_test
    non_recurring_list_train = model.non_recurring_list
    test_dataloader = model.test_dataloader()
    feats_list, probs_list, labels_list, failure_list = [], [], [], []
    for batch_data in test_dataloader:
        features, labels, failure_ids, graphs = batch_data
        feats: th.Tensor = model.module.get_agg_feat(features, graphs)
        probs: th.Tensor = model.forward(features, graphs)
        feats_list.append(feats.detach())
        probs_list.append(probs.detach())
        labels_list.append(labels)
        failure_list.append(failure_ids)
    failure_list = th.concat(failure_list).tolist()
    feats_list = th.concat(feats_list)
    logger.debug(f"{feats_list.shape=}")
    probs = th.concat(probs_list)
    labels = th.concat(labels_list)
    probs_list = [list(1./(1. + np.exp([-prob for prob in prob_tmp]))) for prob_tmp in probs.tolist()]
    preds_list: List[List] = th.argsort(probs, dim=-1, descending=True).tolist()
    feats_list = th.concat([feats[None,preds[0],:] for feats, preds in zip(feats_list, preds_list)])
    labels_list = [set(th.where(label >= 1)[0].tolist()) for label in labels]

    keep_indices = [i for i, failure in enumerate(failure_list) if failure not in non_recurring_list_train]
    failure_list = [failure_list[i] for i in keep_indices]
    feats_list = th.stack([feats_list[i] for i in keep_indices])
    probs_list = [probs_list[i] for i in keep_indices]
    preds_list = [preds_list[i] for i in keep_indices]
    labels_list = [labels_list[i] for i in keep_indices]

    test_feats_reduced = pca.transform(feats_list)
    x_pic = np.concatenate((feats_reduced, test_feats_reduced))
    is_recurs, is_corrects = [], []
    l1_distances, l2_distances = [], []
    
    predicts, non_recur_labels = [], []
    logger.info(
            f"|{'id':4}|{'FR':<3}|{'AR':<3}|{'recurring':<15}|"
            f"{'root cause':<20}|{'rank-1':<20}|{'rank-2':<20}|{'rank-3':<20}|"
        )
    failure_num = len(failure_list)
    assert (len(feats_list), len(preds_list), len(labels_list), len(probs_list)) == (failure_num, failure_num, failure_num, failure_num), f"failure num inconsistent, {(len(failure_list), len(feats_list), len(preds_list), len(labels_list), len(probs_list))=}"
    for fault_id, feats, preds, labels, probs in zip(failure_list, feats_list, preds_list, labels_list, probs_list):
        is_correct = preds[0] in labels
        probs.sort(reverse=True)
        rc_type = cdp.instance_to_class(cdp.gid_to_instance(preds[0]))
        if rc_type not in centers.keys():
            l1_distance, l2_distance = th.tensor(th.inf), th.tensor(th.inf)
            non_recur = 1
        else:
            l1_distance = th.norm(feats-centers[rc_type], p=1, dim=-1).item()
            l2_distance = th.norm(feats-centers[rc_type], p=2, dim=-1).item()
            non_recur = 1 if l2_distance > rads[rc_type] else 0
        if fault_id in non_recurring_list or fault_id in non_recurring_list_train:
            non_recur_label = 1
            is_recurs.append('non-recur')
            ranks = get_rank(labels, preds, cdp.n_failure_instances)
            logger.info(
                f"|{fault_id:<4.0f}|"
                f"{min(ranks):3.0f}|"
                f"{sum(ranks) / len(ranks):3.0f}|"
                f"{'non recurring'!s:<15}|"
                f"{'':<20}|"
                f"{cdp.gid_to_instance(preds[0]):<20}|"
                f"{cdp.gid_to_instance(preds[1]):<20}|"
                f"{cdp.gid_to_instance(preds[2]):<20}|"
            )
            plot_rc_types.append('non-recur')
            if non_recur:
                color_pic.append(4)
            else:
                color_pic.append(5)
        else:
            non_recur_label = 0
            is_recurs.append('recur')
            ranks = get_rank(labels, preds, cdp.n_failure_instances)
            logger.info(
                f"|{fault_id:<4.0f}|"
                f"{min(ranks):3.0f}|"
                f"{sum(ranks) / len(ranks):3.0f}|"
                f"{'recurring'!s:<15}|"
                f"{','.join([cdp.gid_to_instance(_) for _ in labels]):<20}|"
                f"{cdp.gid_to_instance(preds[0]):<20}|"
                f"{cdp.gid_to_instance(preds[1]):<20}|"
                f"{cdp.gid_to_instance(preds[2]):<20}|"
            )
            plot_rc_types.append(cdp.instance_to_class(cdp.gid_to_instance(preds[0] if is_correct else list(labels)[0])))
            if non_recur:
                color_pic.append(3)
            else:
                color_pic.append(1 if is_correct else 2)
        is_corrects.append('True' if is_correct else 'False')
        l1_distances.append(l1_distance)
        l2_distances.append(l2_distance)
        predicts.append(non_recur)
        non_recur_labels.append(non_recur_label)

    dist_dict = {}
    dist_dict['failure_type'], dist_dict['correct'] = is_recurs, is_corrects
    dist_dict['l1-dist'], dist_dict['l2-dist'] = l1_distances, l1_distances
    dist_df = pd.DataFrame(dist_dict)
    dist_df.to_csv(output_dir / 'mhgl_recur_dist.csv', index=False)
    json_dict = {}
    json_dict['failure_type'], json_dict['correct'] = is_recurs, is_corrects
    json_dict['feats'], json_dict['test_feats'] = feats_cluster.tolist(), feats_list.tolist()

    logger.debug(f"{non_recur_labels=}, {predicts=}")
    precision, recall, fscore, support = precision_recall_fscore_support(non_recur_labels, predicts, beta=beta, average='binary')
    non_recur_labels_new, predicts_new = [], []
    for i in range(len(non_recur_labels)):
        if non_recur_labels[i] == 1:
            non_recur_labels_new.append(1)
            predicts_new.append(predicts[i])
        elif is_corrects[i] == 'True':
            non_recur_labels_new.append(0)
            predicts_new.append(predicts[i])
    logger.debug(f"{non_recur_labels_new=}, {predicts_new=}")
    precision2, recall2, fscore2, support2 = precision_recall_fscore_support(non_recur_labels_new, predicts_new, beta=beta, average='binary')
    result_log = logger.add(output_dir / f'mhgl_f{beta}_score.txt')
    logger.info(
        f"include recurring and wrong-predicted ones\nfscore: {fscore}, precision: {precision}, recall: {recall}\n"
        f"exclude recurring and wrong-predicted ones\nfscore: {fscore2}, precision: {precision2}, recall: {recall2}\n"
    )
    logger.remove(result_log)
    with open(output_dir / 'mhgl_feats.json', 'w') as f:
        json.dump(json_dict, f)
    
    plot_scatter(clusters, color_pic, plot_rc_types, x_pic, cluster_cs_pic, cluster_rs_pic, output_dir / 'mhgl_feature_distribution.pdf')


def detect_nonrecur_contrastive(model: DejaVuModelInterface, output_dir: Path, beta=1.):
    output_dir = output_dir
    output_dir.mkdir(exist_ok=True)
    cdp: FDG = model.train_dataset.cdp
    failure_dataloader = model.train_dataloader()
    feats_list, labels_list = [], []
    for batch_data in failure_dataloader:
        features, labels, failure_ids, graphs = batch_data
        feats: th.Tensor = model.module.get_agg_feat(features, graphs)
        valid_idx = th.where(th.any(labels, dim=1))[0]
        feats_list.append(feats.detach()[valid_idx])
        labels_list.append(labels.detach()[valid_idx])
    feats_cluster = th.concat(feats_list)
    logger.debug(f"{feats_cluster.shape=}")
    labels_list = [set(th.where(label >= 1)[0].tolist()) for label in th.concat(labels_list)]
    clusters = DefaultDict(list)
    i = 0
    feats_cluster_ext = []
    for feats, labels in zip(feats_cluster, labels_list):
        # one instance can be delivered to multiple clusters.
        for label in labels:
            rc_type = cdp.instance_to_class(cdp.gid_to_instance(label))
            feats_cluster_ext.append(feats[None,label,:])
            clusters[rc_type].append(i)
            i += 1
    feats_cluster = th.concat(feats_cluster_ext)
    centers = {}
    rads = {}
    for cluster, value in clusters.items():
        center = th.mean(feats_cluster[value], dim=0)
        dists = th.sigmoid(func.pairwise_distance(feats_cluster[value], center))
        rad = th.max(dists)
        centers[cluster] = center
        rads[cluster] = rad.item()

    pca = PCA(n_components=2)
    pca.fit(feats_cluster)
    feats_reduced = pca.transform(feats_cluster)
    color_pic = [0 for _ in range(feats_reduced.shape[0])]
    plot_rc_types = ['' for _ in range(feats_reduced.shape[0])]
    cluster_cs_pic = []
    cluster_rs_pic = []
    for cluster, value in clusters.items():
        tmp = feats_reduced[value, :]
        cluster_c_pic = np.mean(tmp, axis=0)
        tmp_d = np.linalg.norm(tmp-cluster_c_pic, ord=2, axis=-1)
        rad = np.max(tmp_d)
        cluster_cs_pic.append(cluster_c_pic)
        cluster_rs_pic.append(rad)

    non_recurring_list = model.non_recurring_list_test
    non_recurring_list_train = model.non_recurring_list
    test_dataloader = model.test_dataloader()
    feats_list, probs_list, labels_list, failure_list = [], [], [], []
    for batch_data in test_dataloader:
        features, labels, failure_ids, graphs = batch_data
        feats: th.Tensor = model.module.get_agg_feat(features, graphs)
        probs: th.Tensor = model.forward(features, graphs)
        feats_list.append(feats.detach())
        probs_list.append(probs.detach())
        labels_list.append(labels)
        failure_list.append(failure_ids)
    failure_list = th.concat(failure_list).tolist()
    feats_list = th.concat(feats_list)
    logger.debug(f"{feats_list.shape=}")
    probs = th.concat(probs_list)
    labels = th.concat(labels_list)
    probs_list = [list(1./(1. + np.exp([-prob for prob in prob_tmp]))) for prob_tmp in probs.tolist()]
    preds_list: List[List] = th.argsort(probs, dim=-1, descending=True).tolist()
    feats_list = th.concat([feats[None,preds[0],:] for feats, preds in zip(feats_list, preds_list)])
    labels_list = [set(th.where(label >= 1)[0].tolist()) for label in labels]

    keep_indices = [i for i, failure in enumerate(failure_list) if failure not in non_recurring_list_train]
    failure_list = [failure_list[i] for i in keep_indices]
    feats_list = th.stack([feats_list[i] for i in keep_indices])
    probs_list = [probs_list[i] for i in keep_indices]
    preds_list = [preds_list[i] for i in keep_indices]
    labels_list = [labels_list[i] for i in keep_indices]

    test_feats_reduced = pca.transform(feats_list)
    x_pic = np.concatenate((feats_reduced, test_feats_reduced))
    is_recurs, is_corrects = [], []
    distances = []
    
    predicts, non_recur_labels = [], []
    logger.info(
            f"|{'id':4}|{'FR':<3}|{'AR':<3}|{'recurring':<15}|"
            f"{'root cause':<20}|{'rank-1':<20}|{'rank-2':<20}|{'rank-3':<20}|"
        )
    failure_num = len(failure_list)
    assert (len(feats_list), len(preds_list), len(labels_list), len(probs_list)) == (failure_num, failure_num, failure_num, failure_num), f"failure num inconsistent, {(len(failure_list), len(feats_list), len(preds_list), len(labels_list), len(probs_list))=}"
    for fault_id, feats, preds, labels, probs in zip(failure_list, feats_list, preds_list, labels_list, probs_list):
        is_correct = preds[0] in labels
        probs.sort(reverse=True)
        rc_type = rc_type = cdp.instance_to_class(cdp.gid_to_instance(preds[0]))
        if rc_type not in centers.keys():
            distance = th.tensor(th.inf)
            non_recur = 1
        else:
            distance = th.sigmoid(func.pairwise_distance(feats, centers[rc_type])).item()
            non_recur = 1 if distance > rads[rc_type] else 0
        if fault_id in non_recurring_list or fault_id in non_recurring_list_train:
            non_recur_label = 1
            is_recurs.append('non-recur')
            ranks = get_rank(labels, preds, cdp.n_failure_instances)
            logger.info(
                f"|{fault_id:<4.0f}|"
                f"{min(ranks):3.0f}|"
                f"{sum(ranks) / len(ranks):3.0f}|"
                f"{'non recurring'!s:<15}|"
                f"{'':<20}|"
                f"{cdp.gid_to_instance(preds[0]):<20}|"
                f"{cdp.gid_to_instance(preds[1]):<20}|"
                f"{cdp.gid_to_instance(preds[2]):<20}|"
            )
            plot_rc_types.append('non-recur')
            if non_recur:
                color_pic.append(4)
            else:
                color_pic.append(5)
        else:
            non_recur_label = 0
            is_recurs.append('recur')
            ranks = get_rank(labels, preds, cdp.n_failure_instances)
            logger.info(
                f"|{fault_id:<4.0f}|"
                f"{min(ranks):3.0f}|"
                f"{sum(ranks) / len(ranks):3.0f}|"
                f"{'recurring'!s:<15}|"
                f"{','.join([cdp.gid_to_instance(_) for _ in labels]):<20}|"
                f"{cdp.gid_to_instance(preds[0]):<20}|"
                f"{cdp.gid_to_instance(preds[1]):<20}|"
                f"{cdp.gid_to_instance(preds[2]):<20}|"
            )
            plot_rc_types.append(cdp.instance_to_class(cdp.gid_to_instance(preds[0] if is_correct else list(labels)[0])))
            if non_recur:
                color_pic.append(3)
            else:
                color_pic.append(1 if is_correct else 2)
        is_corrects.append('True' if is_correct else 'False')
        distances.append(distance)
        predicts.append(non_recur)
        non_recur_labels.append(non_recur_label)

    dist_dict = {}
    dist_dict['failure_type'], dist_dict['correct'] = is_recurs, is_corrects
    dist_dict['dist'] = distances
    dist_df = pd.DataFrame(dist_dict)
    dist_df.to_csv(output_dir / 'dejavu_omni_wo_nfd_recur_dist.csv', index=False)
    json_dict = {}
    json_dict['failure_type'], json_dict['correct'] = is_recurs, is_corrects
    json_dict['feats'], json_dict['test_feats'] = feats_cluster.tolist(), feats_list.tolist()

    logger.debug(f"{non_recur_labels=}, {predicts=}")
    precision, recall, fscore, support = precision_recall_fscore_support(non_recur_labels, predicts, beta=beta, average='binary')
    non_recur_labels_new, predicts_new = [], []
    for i in range(len(non_recur_labels)):
        if non_recur_labels[i] == 1:
            non_recur_labels_new.append(1)
            predicts_new.append(predicts[i])
        elif is_corrects[i] == 'True':
            non_recur_labels_new.append(0)
            predicts_new.append(predicts[i])
    logger.debug(f"{non_recur_labels_new=}, {predicts_new=}")
    precision2, recall2, fscore2, support2 = precision_recall_fscore_support(non_recur_labels_new, predicts_new, beta=beta, average='binary')
    result_log = logger.add(output_dir / f'dejavu_omni_wo_nfd_f{beta}_score.txt')
    logger.info(
        f"include recurring and wrong-predicted ones\nfscore: {fscore}, precision: {precision}, recall: {recall}\n"
        f"exclude recurring and wrong-predicted ones\nfscore: {fscore2}, precision: {precision2}, recall: {recall2}\n"
    )
    logger.remove(result_log)
    with open(output_dir / 'dejavu_omni_wo_nfd_feats.json', 'w') as f:
        json.dump(json_dict, f)
    
    plot_scatter(clusters, color_pic, plot_rc_types, x_pic, cluster_cs_pic, cluster_rs_pic, output_dir / 'dejavu_omni_wo_nfd_feature_distribution.pdf')



def interpret_feat_cluster(model: DejaVuModelInterface, threshold, output_dir: Path, beta=1.):
    output_dir = output_dir
    output_dir.mkdir(exist_ok=True)
    cdp: FDG = model.train_dataset.cdp
    failure_dataloader = model.train_dataloader()
    feats_list, labels_list = [], []
    for batch_data in failure_dataloader:
        features, labels, failure_ids, graphs = batch_data
        feats: th.Tensor = model.module.get_agg_feat(features, graphs)
        valid_idx = th.where(th.any(labels, dim=1))[0]
        feats_list.append(feats.detach()[valid_idx])
        labels_list.append(labels.detach()[valid_idx])
    feats_cluster = th.concat(feats_list)
    logger.debug(f"{feats_cluster.shape=}")
    labels_list = [set(th.where(label >= 1)[0].tolist()) for label in th.concat(labels_list)]
    clusters = DefaultDict(list)
    i = 0
    feats_cluster_ext = []
    for feats, labels in zip(feats_cluster, labels_list):
        # one instance can be delivered to multiple clusters.
        for label in labels:
            rc_type = cdp.instance_to_class(cdp.gid_to_instance(label))
            feats_cluster_ext.append(feats[None,label,:])
            clusters[rc_type].append(i)
            i += 1
    feats_cluster = th.concat(feats_cluster_ext)
    centers = {}
    rads = {}
    for cluster, value in clusters.items():
        center = th.mean(feats_cluster[value], dim=0)
        dists = th.norm(feats_cluster[value]-center, p=2, dim=-1)
        rad = th.max(dists)
        centers[cluster] = center
        rads[cluster] = rad.item()

    pca = PCA(n_components=2)
    pca.fit(feats_cluster)
    feats_reduced = pca.transform(feats_cluster)
    color_pic = [0 for _ in range(feats_reduced.shape[0])]
    plot_rc_types = ['' for _ in range(feats_reduced.shape[0])]
    cluster_cs_pic = []
    cluster_rs_pic = []
    for cluster, value in clusters.items():
        tmp = feats_reduced[value, :]
        cluster_c_pic = np.mean(tmp, axis=0)
        tmp_d = np.linalg.norm(tmp-cluster_c_pic, ord=2, axis=-1)
        rad = np.max(tmp_d)
        cluster_cs_pic.append(cluster_c_pic)
        cluster_rs_pic.append(rad)

    non_recurring_list = model.non_recurring_list_test
    non_recurring_list_train = model.non_recurring_list
    test_dataloader = model.test_dataloader()
    feats_list, probs_list, labels_list, failure_list = [], [], [], []
    for batch_data in test_dataloader:
        features, labels, failure_ids, graphs = batch_data
        feats: th.Tensor = model.module.get_agg_feat(features, graphs)
        probs: th.Tensor = model.forward(features, graphs)
        feats_list.append(feats.detach())
        probs_list.append(probs.detach())
        labels_list.append(labels)
        failure_list.append(failure_ids)
    failure_list = th.concat(failure_list).tolist()
    feats_list = th.concat(feats_list)
    logger.debug(f"{feats_list.shape=}")
    probs = th.concat(probs_list)
    labels = th.concat(labels_list)
    probs_list = [list(1./(1. + np.exp([-prob for prob in prob_tmp]))) for prob_tmp in probs.tolist()]
    preds_list: List[List] = th.argsort(probs, dim=-1, descending=True).tolist()
    feats_list = th.concat([feats[None,preds[0],:] for feats, preds in zip(feats_list, preds_list)])
    labels_list = [set(th.where(label >= 1)[0].tolist()) for label in labels]
    keep_indices = [i for i, failure in enumerate(failure_list) if failure not in non_recurring_list_train]
    failure_list = [failure_list[i] for i in keep_indices]
    feats_list = th.stack([feats_list[i] for i in keep_indices])
    probs_list = [probs_list[i] for i in keep_indices]
    preds_list = [preds_list[i] for i in keep_indices]
    labels_list = [labels_list[i] for i in keep_indices]

    test_feats_reduced = pca.transform(feats_list)
    x_pic = np.concatenate((feats_reduced, test_feats_reduced))
    is_recurs, is_corrects = [], []
    l1_distances, l2_distances = [], []
    
    predicts, predicts_score, non_recur_labels = [], [], []
    logger.info(
            f"|{'id':4}|{'FR':<3}|{'AR':<3}|{'recurring':<15}|"
            f"{'root cause':<20}|{'rank-1':<20}|{'rank-2':<20}|{'rank-3':<20}|"
        )
    failure_num = len(failure_list)
    assert (len(feats_list), len(preds_list), len(labels_list), len(probs_list)) == (failure_num, failure_num, failure_num, failure_num), f"failure num inconsistent, {(len(failure_list), len(feats_list), len(preds_list), len(labels_list), len(probs_list))=}"
    for fault_id, feats, preds, labels, probs in zip(failure_list, feats_list, preds_list, labels_list, probs_list):
        is_correct = preds[0] in labels
        probs.sort(reverse=True)
        non_recur_score = probs[0] < threshold
        rc_type = rc_type = cdp.instance_to_class(cdp.gid_to_instance(preds[0]))
        if rc_type not in centers.keys():
            l1_distance, l2_distance = th.tensor(th.inf), th.tensor(th.inf)
            non_recur = 1
        else:
            l1_distance = th.norm(feats-centers[rc_type], p=1, dim=-1).item()
            l2_distance = th.norm(feats-centers[rc_type], p=2, dim=-1).item()
            non_recur = 1 if l2_distance > rads[rc_type] else 0
        if fault_id in non_recurring_list or fault_id in non_recurring_list_train:
            non_recur_label = 1
            is_recurs.append('non-recur')
            ranks = get_rank(labels, preds, cdp.n_failure_instances)
            logger.info(
                f"|{fault_id:<4.0f}|"
                f"{min(ranks):3.0f}|"
                f"{sum(ranks) / len(ranks):3.0f}|"
                f"{'non recurring'!s:<15}|"
                f"{'':<20}|"
                f"{cdp.gid_to_instance(preds[0]):<20}|"
                f"{cdp.gid_to_instance(preds[1]):<20}|"
                f"{cdp.gid_to_instance(preds[2]):<20}|"
            )
            plot_rc_types.append('non-recur')
            if non_recur_score:
                color_pic.append(4)
            else:
                color_pic.append(5)
        else:
            non_recur_label = 0
            is_recurs.append('recur')
            ranks = get_rank(labels, preds, cdp.n_failure_instances)
            logger.info(
                f"|{fault_id:<4.0f}|"
                f"{min(ranks):3.0f}|"
                f"{sum(ranks) / len(ranks):3.0f}|"
                f"{'recurring'!s:<15}|"
                f"{','.join([cdp.gid_to_instance(_) for _ in labels]):<20}|"
                f"{cdp.gid_to_instance(preds[0]):<20}|"
                f"{cdp.gid_to_instance(preds[1]):<20}|"
                f"{cdp.gid_to_instance(preds[2]):<20}|"
            )
            plot_rc_types.append(cdp.instance_to_class(cdp.gid_to_instance(preds[0] if is_correct else list(labels)[0])))
            if non_recur_score:
                color_pic.append(3)
            else:
                color_pic.append(1 if is_correct else 2)
        is_corrects.append('True' if is_correct else 'False')
        l1_distances.append(l1_distance)
        l2_distances.append(l2_distance)
        predicts.append(non_recur)
        predicts_score.append(non_recur_score)
        non_recur_labels.append(non_recur_label)

    dist_dict = {}
    dist_dict['failure_type'], dist_dict['correct'] = is_recurs, is_corrects
    dist_dict['l1-dist'], dist_dict['l2-dist'] = l1_distances, l1_distances
    dist_df = pd.DataFrame(dist_dict)
    dist_df.to_csv(output_dir / 'interpret_recur_dist.csv', index=False)
    json_dict = {}
    json_dict['failure_type'], json_dict['correct'] = is_recurs, is_corrects
    json_dict['feats'], json_dict['test_feats'] = feats_cluster.tolist(), feats_list.tolist()

    logger.debug(f"{non_recur_labels=}, {predicts=}")
    precision, recall, fscore, support = precision_recall_fscore_support(non_recur_labels, predicts_score, beta=beta, average='binary')
    non_recur_labels_new, predicts_new = [], []
    for i in range(len(non_recur_labels)):
        if non_recur_labels[i] == 1:
            non_recur_labels_new.append(1)
            predicts_new.append(predicts_score[i])
        elif is_corrects[i] == 'True':
            non_recur_labels_new.append(0)
            predicts_new.append(predicts_score[i])
    logger.debug(f"{non_recur_labels_new=}, {predicts_new=}")
    precision2, recall2, fscore2, support2 = precision_recall_fscore_support(non_recur_labels_new, predicts_new, beta=beta, average='binary')
    result_log = logger.add(output_dir / f'interpret_f{beta}_score.txt')
    logger.info(
        f"include recurring and wrong-predicted ones\nfscore: {fscore}, precision: {precision}, recall: {recall}\n"
        f"exclude recurring and wrong-predicted ones\nfscore: {fscore2}, precision: {precision2}, recall: {recall2}\n"
    )
    logger.remove(result_log)
    with open(output_dir / 'interpret_recur_feats.json', 'w') as f:
        json.dump(json_dict, f)
    
    plot_scatter(clusters, color_pic, plot_rc_types, x_pic, cluster_cs_pic, cluster_rs_pic, output_dir / 'interpret_feature_distribution.pdf')


def detect_nonrecur_GMM(model: DejaVuModelInterface, output_dir: Path, beta=1.):
    output_dir = output_dir
    output_dir.mkdir(exist_ok=True)
    cdp: FDG = model.train_dataset.cdp
    failure_dataloader = model.train_dataloader()
    feats_list, labels_list = [], []
    for batch_data in failure_dataloader:
        features, labels, failure_ids, graphs = batch_data
        feats: th.Tensor = model.module.get_agg_feat(features, graphs)
        valid_idx = th.where(th.any(labels, dim=1))[0]
        feats_list.append(feats.detach()[valid_idx])
        labels_list.append(labels.detach()[valid_idx])
    feats_cluster = th.concat(feats_list)
    labels_list = [set(th.where(label >= 1)[0].tolist()) for label in th.concat(labels_list)]
    clusters = set()
    feats_cluster_ext = []
    for feats, labels in zip(feats_cluster, labels_list):
        # one instance can be delivered to multiple clusters.
        for label in labels:
            rc_type = cdp.instance_to_class(cdp.gid_to_instance(label))
            feats_cluster_ext.append(feats[None,label,:])
            clusters.add(rc_type)
    feats_cluster = th.concat(feats_cluster_ext)
    n_components = len(clusters)
    logger.debug(f"{n_components=}")
    X_train_np = feats_cluster.cpu().numpy()
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(X_train_np)
    probs_train = gmm.score_samples(X_train_np)
    threshold = np.percentile(probs_train, 5)

    non_recurring_list = model.non_recurring_list_test
    non_recurring_list_train = model.non_recurring_list
    test_dataloader = model.test_dataloader()
    feats_list, probs_list, labels_list, failure_list = [], [], [], []
    for batch_data in test_dataloader:
        features, labels, failure_ids, graphs = batch_data
        feats: th.Tensor = model.module.get_agg_feat(features, graphs)
        probs: th.Tensor = model.forward(features, graphs)
        feats_list.append(feats.detach())
        probs_list.append(probs.detach())
        labels_list.append(labels)
        failure_list.append(failure_ids)
    failure_list = th.concat(failure_list).tolist()
    feats_list = th.concat(feats_list)
    logger.debug(f"{feats_list.shape=}")
    probs = th.concat(probs_list)
    labels = th.concat(labels_list)
    probs_list = [list(1./(1. + np.exp([-prob for prob in prob_tmp]))) for prob_tmp in probs.tolist()]
    preds_list: List[List] = th.argsort(probs, dim=-1, descending=True).tolist()
    feats_list = th.concat([feats[None,preds[0],:] for feats, preds in zip(feats_list, preds_list)])
    labels_list = [set(th.where(label >= 1)[0].tolist()) for label in labels]

    keep_indices = [i for i, failure in enumerate(failure_list) if failure not in non_recurring_list_train]
    failure_list = [failure_list[i] for i in keep_indices]
    feats_list = th.stack([feats_list[i] for i in keep_indices])
    probs_list = [probs_list[i] for i in keep_indices]
    preds_list = [preds_list[i] for i in keep_indices]
    labels_list = [labels_list[i] for i in keep_indices]

    predicts, non_recur_labels = [], []
    failure_num = len(failure_list)
    assert (len(feats_list), len(preds_list), len(labels_list), len(probs_list)) == (failure_num, failure_num, failure_num, failure_num), f"failure num inconsistent, {(len(failure_list), len(feats_list), len(preds_list), len(labels_list), len(probs_list))=}"
    for fault_id, feats in zip(failure_list, feats_list):
        prob = gmm.score_samples(feats[None, :].cpu().numpy())
        non_recur = 1 if prob < threshold else 0
        if fault_id in non_recurring_list or fault_id in non_recurring_list_train:
            non_recur_label = 1
        else:
            non_recur_label = 0
        predicts.append(non_recur)
        non_recur_labels.append(non_recur_label)

    logger.debug(f"{non_recur_labels=}, {predicts=}")
    precision, recall, fscore, support = precision_recall_fscore_support(non_recur_labels, predicts, beta=beta, average='binary')
    result_log = logger.add(output_dir / f'gmm_f{beta}_score.txt')
    logger.info(
        f"include recurring and wrong-predicted ones\nfscore: {fscore}, precision: {precision}, recall: {recall}\n"
    )
    logger.remove(result_log)


def detect_nonrecur_Kmeans(model: DejaVuModelInterface, output_dir: Path, beta=1.):
    output_dir = output_dir
    output_dir.mkdir(exist_ok=True)
    cdp: FDG = model.train_dataset.cdp
    failure_dataloader = model.train_dataloader()
    feats_list, labels_list = [], []
    for batch_data in failure_dataloader:
        features, labels, failure_ids, graphs = batch_data
        feats: th.Tensor = model.module.get_agg_feat(features, graphs)
        valid_idx = th.where(th.any(labels, dim=1))[0]
        feats_list.append(feats.detach()[valid_idx])
        labels_list.append(labels.detach()[valid_idx])
    feats_cluster = th.concat(feats_list)
    logger.debug(f"{feats_cluster.shape=}")
    labels_list = [set(th.where(label >= 1)[0].tolist()) for label in th.concat(labels_list)]
    clusters = set()
    feats_cluster_ext = []
    for feats, labels in zip(feats_cluster, labels_list):
        # one instance can be delivered to multiple clusters.
        for label in labels:
            rc_type = cdp.instance_to_class(cdp.gid_to_instance(label))
            feats_cluster_ext.append(feats[None,label,:])
            clusters.add(rc_type)
    feats_cluster = th.concat(feats_cluster_ext)
    n_clusters = len(clusters)
    logger.debug(f"{n_clusters=}")
    X_train_np = feats_cluster.cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_train_np)
    cluster_radii = []
    for i in range(kmeans.n_clusters):
        cluster_points = X_train_np[kmeans.labels_ == i]
        cluster_center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        cluster_radius = np.max(distances)
        cluster_radii.append(cluster_radius)

    non_recurring_list = model.non_recurring_list_test
    non_recurring_list_train = model.non_recurring_list
    test_dataloader = model.test_dataloader()
    feats_list, probs_list, labels_list, failure_list = [], [], [], []
    for batch_data in test_dataloader:
        features, labels, failure_ids, graphs = batch_data
        feats: th.Tensor = model.module.get_agg_feat(features, graphs)
        probs: th.Tensor = model.forward(features, graphs)
        feats_list.append(feats.detach())
        probs_list.append(probs.detach())
        labels_list.append(labels)
        failure_list.append(failure_ids)
    failure_list = th.concat(failure_list).tolist()
    feats_list = th.concat(feats_list)
    logger.debug(f"{feats_list.shape=}")
    probs = th.concat(probs_list)
    labels = th.concat(labels_list)
    probs_list = [list(1./(1. + np.exp([-prob for prob in prob_tmp]))) for prob_tmp in probs.tolist()]
    preds_list: List[List] = th.argsort(probs, dim=-1, descending=True).tolist()
    feats_list = th.concat([feats[None,preds[0],:] for feats, preds in zip(feats_list, preds_list)])
    labels_list = [set(th.where(label >= 1)[0].tolist()) for label in labels]

    keep_indices = [i for i, failure in enumerate(failure_list) if failure not in non_recurring_list_train]
    failure_list = [failure_list[i] for i in keep_indices]
    feats_list = th.stack([feats_list[i] for i in keep_indices])
    probs_list = [probs_list[i] for i in keep_indices]
    preds_list = [preds_list[i] for i in keep_indices]
    labels_list = [labels_list[i] for i in keep_indices]

    predicts, non_recur_labels = [], []
    failure_num = len(failure_list)
    assert (len(feats_list), len(preds_list), len(labels_list), len(probs_list)) == (failure_num, failure_num, failure_num, failure_num), f"failure num inconsistent, {(len(failure_list), len(feats_list), len(preds_list), len(labels_list), len(probs_list))=}"
    for fault_id, feats in zip(failure_list, feats_list):
        distance = np.linalg.norm(kmeans.cluster_centers_ - feats[None, :].cpu().numpy(), axis=1)
        nearest_cluster_index = np.argmin(distance)
        nearest_cluster_radius = cluster_radii[nearest_cluster_index]
        non_recur = 1 if distance[nearest_cluster_index] > nearest_cluster_radius else 0
        if fault_id in non_recurring_list or fault_id in non_recurring_list_train:
            non_recur_label = 1
        else:
            non_recur_label = 0
        predicts.append(non_recur)
        non_recur_labels.append(non_recur_label)

    logger.debug(f"{non_recur_labels=}, {predicts=}")
    precision, recall, fscore, support = precision_recall_fscore_support(non_recur_labels, predicts, beta=beta, average='binary')
    result_log = logger.add(output_dir / f'kmeans_f{beta}_score.txt')
    logger.info(
        f"include recurring and wrong-predicted ones\nfscore: {fscore}, precision: {precision}, recall: {recall}\n"
    )
    logger.remove(result_log)
