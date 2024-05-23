import torch as th
import torch.jit
import torch.nn.functional as func
import random
from failure_dependency_graph import FDG
from typing import DefaultDict, List


# @torch.jit.script
def binary_classification_loss(
        prob: th.Tensor, label: th.Tensor, gamma: float = 0.,
        normal_fault_weight: float = 1e-1,
        target_node_weight: float = 0.5,
) -> th.Tensor:
    assert prob.size() == label.size()
    device = prob.device
    target_id = th.argmax(label, dim=-1)
    prob_softmax = th.sigmoid(prob)
    weights = th.pow(
        th.abs(label - prob_softmax), gamma
    ) * (
                      th.max(label, dim=-1, keepdim=True).values + th.full_like(label, fill_value=normal_fault_weight)
              )
    if len(prob.size()) == 1:
        weights[target_id] *= prob.size()[-1] * target_node_weight
    else:
        weights[th.arange(len(target_id), device=device), target_id] *= prob.size()[-1] * target_node_weight
    loss = func.binary_cross_entropy_with_logits(prob, label.float(), reduction='none')
    return th.sum(loss * weights) / th.prod(th.tensor(weights.size()))


def contrastive_loss(feat: th.Tensor, label: th.Tensor, cdp: FDG, w: float, n_samples: int) -> th.Tensor:
    def pair_loss(feature1, feature2, label):
        distance = th.sigmoid(func.pairwise_distance(feature1, feature2))
        loss_same = label * th.pow(distance, 2)
        loss_diff = (1 - label) * th.pow(th.clamp(1 - distance, min=0.0), 2)
        loss_contrastive = th.mean(loss_same + loss_diff)
        return loss_contrastive

    valid_idx = th.where(th.any(label, dim=1))[0]
    labels_list = [set(th.where(l >= 1)[0].tolist()) for l in label[valid_idx]]
    clusters = DefaultDict(list)
    feats_cluster_ext = []
    i = 0
    for feats, labels in zip(feat, labels_list):
        # one instance can be delivered to multiple clusters.
        for label in labels:
            rc_type = cdp.instance_to_class(cdp.gid_to_instance(label))
            feats_cluster_ext.append(th.mean(feats, dim=0)[None, :])
            clusters[rc_type].append(i)
            i += 1
    feats_cluster = th.concat(feats_cluster_ext)

    categories = list(clusters.keys())
    positive_pairs, negative_pairs = [], []
    n_positive = n_samples // 2
    n_negative = n_samples - n_positive
    for _ in range(n_positive):
        category = random.choice(categories)
        if len(clusters[category]) > 1:
            sample1, sample2 = random.sample(clusters[category], 2)
            positive_pairs.append((th.mean(feats_cluster[sample1], dim=0), th.mean(feats_cluster[sample2], dim=0), 1))
    if len(categories) > 1:
        for _ in range(n_negative):
            category1, category2 = random.sample(categories, 2)
            sample1 = random.choice(clusters[category1])
            sample2 = random.choice(clusters[category2])
            negative_pairs.append((th.mean(feats_cluster[sample1], dim=0), th.mean(feats_cluster[sample2], dim=0), 0))
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    feature1_list = [pair[0] for pair in all_pairs]
    feature2_list = [pair[1] for pair in all_pairs]
    labels_list = [pair[2] for pair in all_pairs]
    feature1 = torch.stack(feature1_list)
    feature2 = torch.stack(feature2_list)
    labels = torch.tensor(labels_list, dtype=torch.float32, device=feature1.device)
    loss = pair_loss(feature1, feature2, labels) * w
    return loss


@torch.jit.script
def focal_loss(preds, labels, gamma: float = 2.):
    preds = preds.view(-1, preds.size(-1))  # [-1, num_classes]
    labels = labels.view(-1, 1)  # [-1, ]
    preds_logsoft = func.log_softmax(preds, dim=-1)  # log_softmax
    preds_softmax = th.exp(preds_logsoft)  # softmax
    preds_softmax = preds_softmax.gather(1, labels)  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
    preds_logsoft = preds_logsoft.gather(1, labels)
    weights = th.pow((1 - preds_softmax), gamma)
    loss = - weights * preds_logsoft  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

    loss = th.sum(loss) / th.sum(weights)
    return loss


@torch.jit.script
def multi_classification_loss(prob: th.Tensor, label: th.Tensor, gamma: float = 2) -> th.Tensor:
    assert prob.size() == label.size()
    target_id = th.argmax(label, dim=-1)
    assert len(prob.size()) == len(target_id.size()) + 1
    if len(prob.size()) == 1:
        return focal_loss(prob.view(1, -1), target_id.view(1), gamma=gamma)
    else:
        return focal_loss(prob, target_id, gamma=gamma)


def cluster_loss(feat: th.Tensor, prob: th.Tensor, label: th.Tensor, cdp: FDG, w1: float, w2: float) -> th.Tensor:
    valid_idx = th.where(th.any(label, dim=1))[0]
    pred_list: List[List] = th.argsort(prob[valid_idx], dim=-1, descending=True).tolist()
    labels_list = [set(th.where(l >= 1)[0].tolist()) for l in label[valid_idx]]
    clusters = DefaultDict(list)

    feats_cluster_ext = []
    i = 0
    for feats, labels in zip(feat, labels_list):
        # one instance can be delivered to multiple clusters.
        for label in labels:
            rc_type = cdp.instance_to_class(cdp.gid_to_instance(label))
            feats_cluster_ext.append(feats[None,label,:])
            clusters[rc_type].append(i)
            i += 1
    feats_cluster = th.concat(feats_cluster_ext)
    centers = []
    dists = []
    for cluster, value in clusters.items():
        center = th.mean(feats_cluster[value], dim=0)
        centers.append(center)
        dists.append(th.norm(center-feats_cluster[value], p=2, dim=1))
    dists = th.concat([dist[None, :] for dist in dists], dim=1)
    loss_in_cluster = th.mean(dists)
    dists = []
    cluster_num = len(centers)
    if cluster_num == 1:
        return loss_in_cluster if loss_in_cluster==0 else w1/loss_in_cluster.detach()*loss_in_cluster
    else:
        i = 0
        for cluster, value in clusters.items():
            for j in range(cluster_num):
                if i != j:
                    dists.append(1.0 / th.norm(feats_cluster[value]-centers[j], p=2, dim=1))
            i += 1
        dists = th.concat([dist[None, dist!=th.inf] for dist in dists], dim=1)
        loss_between_cluster = th.mean(dists)
        loss_in_cluster = loss_in_cluster if loss_in_cluster==0 else w1/loss_in_cluster.detach()*loss_in_cluster
        loss_between_cluster = loss_between_cluster if loss_between_cluster==0 else w2/loss_between_cluster.detach()*loss_between_cluster
    return loss_in_cluster + loss_between_cluster


def feature_cluster_agg_loss(feat: th.Tensor, prob: th.Tensor, label: th.Tensor, cdp: FDG, w1: float, w2: float) -> th.Tensor:
    valid_idx = th.where(th.any(label, dim=1))[0]
    pred_list: List[List] = th.argsort(prob[valid_idx], dim=-1, descending=True).tolist()
    labels_list = [set(th.where(l >= 1)[0].tolist()) for l in label[valid_idx]]
    clusters = DefaultDict(list)

    feats_cluster_ext = []
    i = 0
    for feats, labels in zip(feat, labels_list):
        # one instance can be delivered to multiple clusters.
        for label in labels:
            rc_type = cdp.instance_to_class(cdp.gid_to_instance(label))
            feats_cluster_ext.append(th.mean(feats, dim=0)[None, :])
            clusters[rc_type].append(i)
            i += 1
    feats_cluster = th.concat(feats_cluster_ext)

    centers = []
    dists = []
    for cluster, value in clusters.items():
        center = th.mean(feats_cluster[value], dim=0)
        centers.append(center)
        dists.append(th.norm(center-feats_cluster[value], p=2, dim=1))
    dists = th.concat([dist[None, :] for dist in dists], dim=1)
    loss_in_cluster = th.mean(dists)
    dists = []
    cluster_num = len(centers)
    if cluster_num == 1:
        return loss_in_cluster if loss_in_cluster==0 else w1/loss_in_cluster.detach()*loss_in_cluster
    else:
        i = 0
        for cluster, value in clusters.items():
            for j in range(cluster_num):
                if i != j:
                    dists.append(1.0 / th.norm(feats_cluster[value]-centers[j], p=2, dim=1))
            i += 1
        dists = th.concat([dist[None, dist!=th.inf] for dist in dists], dim=1)
        loss_between_cluster = th.mean(dists)
        loss_in_cluster = loss_in_cluster if loss_in_cluster==0 else w1/loss_in_cluster.detach()*loss_in_cluster
        loss_between_cluster = loss_between_cluster if loss_between_cluster==0 else w2/loss_between_cluster.detach()*loss_between_cluster
    return loss_in_cluster + loss_between_cluster
