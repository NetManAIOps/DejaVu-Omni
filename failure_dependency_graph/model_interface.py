from collections import defaultdict
from functools import cached_property, reduce
from typing import List, Optional, Tuple, TypeVar, Generic, Any, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch as th
from diskcache import Cache
from loguru import logger
from pytorch_lightning.core.mixins import DeviceDtypeModuleMixin
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from failure_dependency_graph.FDG_config import FDGBaseConfig
from failure_dependency_graph.failure_dependency_graph import FDG

__all__ = ["FDGModelInterface", "split_failures_by_type"]

from metric_preprocess import MetricPreprocessor

CONFIG_T = TypeVar('CONFIG_T', bound=FDGBaseConfig)
DATASET_T = TypeVar('DATASET_T', bound=Dataset)


class FDGModelInterface(pl.LightningModule, Generic[CONFIG_T, DATASET_T]):
    def __init__(
            self, config: CONFIG_T, loaded_FDG: Optional[FDG] = None,
    ):
        super().__init__()

        self._config = config
        self._fdg, real_paths = FDG.load(
            dir_path=config.data_dir,
            graph_path=config.graph_config_path, metrics_path=config.metrics_path, faults_path=config.faults_path,
            return_real_path=True,
            use_anomaly_direction_constraint=config.use_anomaly_direction_constraint,
            loaded_FDG=loaded_FDG,
            flush_dataset_cache=config.flush_dataset_cache
        )
        dataset_cache_dir = config.cache_dir / ".".join(
            [
                f"{k}={real_paths[k]}" for k in sorted(real_paths.keys())
            ] + [
                f'use_anomaly_direction_constraint={config.use_anomaly_direction_constraint}',
                f'clip_{config.input_clip_val if config.input_clip_val is not None else "no"}'
            ]
        ).replace("/", "_")
        logger.info(f"dataset_cache_dir={dataset_cache_dir}")
        self._cache = Cache(str(dataset_cache_dir), size_limit=int(1e10))

        if self._config.dataset_split_method == 'type':
            self._train_fault_ids, self._validation_fault_ids, self._test_fault_ids = split_failures_by_type(
                self._fdg.failures_df, split=self._config.dataset_split_ratio,
                train_set_sampling_ratio=self._config.train_set_sampling,
                balance_train_set=self._config.balance_train_set,
                fdg=self.fdg,
            )
            self._non_recurring_list, self._non_recurring_list_test = [], []
            self._drift_list = []
        elif self._config.dataset_split_method == 'recur':
            self._train_fault_ids, self._validation_fault_ids, self._test_fault_ids, self._non_recurring_list, non_recurring_fault_set, self._non_recurring_list_test, non_recurring_fault_set_test = split_failures_by_recur(
                self._fdg.failures_df, split=self._config.dataset_split_ratio,
                train_set_sampling_ratio=self._config.train_set_sampling,
                balance_train_set=self._config.balance_train_set,
                fdg=self.fdg,
                non_recur_index_train=self._config.non_recur_index_train,
                non_recur_index_test=self._config.non_recur_index_test
            )
            self._fdg, _ = FDG.load(
                dir_path=config.data_dir,
                graph_path=config.graph_config_path, metrics_path=config.metrics_path, faults_path=config.faults_path,
                return_real_path=True,
                use_anomaly_direction_constraint=config.use_anomaly_direction_constraint,
                loaded_FDG=None,
                flush_dataset_cache=config.flush_dataset_cache,
                non_recurring_fault_set=non_recurring_fault_set | non_recurring_fault_set_test
            )
            self._drift_list = []
        elif self._config.dataset_split_method == 'drift':
            self._train_fault_ids, self._validation_fault_ids, self._test_fault_ids, self._drift_list = split_failures_by_drift(
                self._fdg.failures_df, split=self._config.dataset_split_ratio,
                train_set_sampling_ratio=self._config.train_set_sampling,
                balance_train_set=self._config.balance_train_set,
                fdg=self.fdg, drift_time=self._config.drift_time,
            )
            self._non_recurring_list, self._non_recurring_list_test = [], []
        self._train_dataset, self._validation_dataset, self._test_dataset = None, None, None  # set by self.setup()

    @property
    def cache(self) -> Cache:
        return self._cache

    @property
    def config(self) -> CONFIG_T:
        return self._config

    @property
    def train_dataset(self) -> DATASET_T:
        return self._train_dataset

    @property
    def validation_dataset(self) -> DATASET_T:
        return self._validation_dataset

    @property
    def test_dataset(self) -> DATASET_T:
        return self._test_dataset

    @property
    def fdg(self) -> FDG:
        return self._fdg

    @property
    def train_failure_ids(self) -> List[int]:
        return self._train_fault_ids

    @property
    def validation_failure_ids(self) -> List[int]:
        return self._validation_fault_ids

    @property
    def test_failure_ids(self) -> List[int]:
        return self._test_fault_ids
    
    @property
    def non_recurring_list(self) -> List[int]:
        return self._non_recurring_list

    @property
    def non_recurring_list_test(self) -> List[int]:
        return self._non_recurring_list_test

    @property
    def drift_list(self) -> List[int]:
        return self._drift_list

    @cached_property
    def metric_preprocessor(self) -> MetricPreprocessor:
        return self.get_metric_preprocessor(self._fdg, self._cache, self._config)

    @staticmethod
    def get_metric_preprocessor(fdg: FDG, cache: Cache, config: FDGBaseConfig) -> MetricPreprocessor:
        fe_cache_key = f"MetricPreprocessor"
        if fe_cache_key not in cache or config.flush_dataset_cache:
            cache.set(fe_cache_key, MetricPreprocessor(fdg=fdg, granularity=60, clip_value=config.input_clip_val))
        else:
            logger.warning("Use cached metric preprocessor")
        mp = cache.get(fe_cache_key)
        return mp

    def get_collate_fn(self, batch_size: int):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        """
        setup the three datasets
        If the dataset implement 'Dataset.to(device)', then the FDGModelInterface will automatically call it
            when the module's device is changed
        :param stage:
        :return:
        """
        raise NotImplementedError

    def move_data_set_to_device_(self):
        """
        1. 改变MetricPreprocessor的位置
        2. 如果dataset实现了to方法， 那么在模型改变位置的时候把dataset的位置也一同改变
        dataset中的to方法是为了把dataset的输出直接放到对应的设备上。
        对于map-style dataset， 如果每个index对应的数据不变， 那么就可以cache起来， 从而不需要每个epoch把数据移动到cuda上。
        :return:
        """
        if self._train_dataset is not None and hasattr(self._train_dataset, 'to'):
            self._train_dataset.to(self.device)
        if self._validation_dataset is not None and hasattr(self._validation_dataset, 'to'):
            self._validation_dataset.to(self.device)
        if self._test_dataset is not None and hasattr(self._test_dataset, "to"):
            self._test_dataset.to(self.device)
        self.metric_preprocessor.to(self.device)

    def to(self, *args: Any, **kwargs: Any) -> "DeviceDtypeModuleMixin":
        ret = super().to(*args, **kwargs)
        self.move_data_set_to_device_()
        return ret

    def cuda(self, device: Optional[Union[th.device, int]] = None) -> "DeviceDtypeModuleMixin":
        ret = super().cuda(device)
        self.move_data_set_to_device_()
        return ret

    def cpu(self) -> "DeviceDtypeModuleMixin":
        ret = super().cpu()
        self.move_data_set_to_device_()
        return ret

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self._train_dataset, batch_size=self._config.batch_size,
            collate_fn=self.get_collate_fn(self._config.batch_size),
            shuffle=True,
            pin_memory=False,
        )

    def train_dataloader_orig(self) -> TRAIN_DATALOADERS:
        """
        The dataloader of the training dataset without shuffling.
        :return:
        """
        return DataLoader(
            self._train_dataset, batch_size=self._config.batch_size,
            collate_fn=self.get_collate_fn(self._config.batch_size),
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self._test_dataset, batch_size=self._config.test_batch_size,
            collate_fn=self.get_collate_fn(self._config.batch_size),
            shuffle=False,
            pin_memory=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self._validation_dataset, batch_size=self._config.test_batch_size,
            collate_fn=self.get_collate_fn(self._config.batch_size),
            shuffle=False,
            pin_memory=False,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()


def split_failures_by_type(
        fault_df: pd.DataFrame, *, fdg: FDG = None, split: Tuple[float, float, float] = (0.3, 0.2, 0.5),
        train_set_sampling_ratio: float = 1.0, balance_train_set: bool = False
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(np.random.randint(0,500))  # the random seed should be fixed
    fault_type_2_id_list = defaultdict(list)
    id_2_root_cause_node = {}
    print(fdg._node_to_class)
    for idx, (_, fault) in enumerate(fault_df.iterrows()):
        # kpis = fault['kpi'] if not pd.isnull(fault['kpi']) else ""
        # 考虑根因KPI来对故障分类的话，有很多类别的故障数量太少（只有一两个）
        if fdg is not None:
            fault_type = tuple(fdg.instance_to_class(_) for _ in fault['root_cause_node'].split(";"))
        else:
            fault_type = tuple(fault['node_type'].split(";"))
        fault_type = list(fault_type)
        fault_type.sort()
        fault_type = tuple(fault_type)
        fault_type_2_id_list[fault_type].append(idx)
        id_2_root_cause_node[idx] = set(fault['root_cause_node'].split(";"))
    logger.info(
        f"fault ids with multiple root causes: "
        f"{[(k, len(v)) for k, v in id_2_root_cause_node.items() if len(v) > 1]}"
    )
    train_ids_list: List[List[int]] = []
    validation_list: List[int] = []
    test_list: List[int] = []
    for fault_type, ids in fault_type_2_id_list.items():
        _train_split = max(int(len(ids) * split[0]), 1)
        _valid_split = max(int(len(ids) * (split[0] + split[1])), 1)
        rng.shuffle(ids)

        train_ids = ids[0:_train_split]
        train_ids_list.append(train_ids)

        validation_ids = ids[_train_split:_valid_split]
        validation_list.extend(validation_ids)

        test_ids = ids[_valid_split:]
        test_list.extend(test_ids)

        del _train_split, _valid_split

        train_rc_nodes = reduce(lambda a, b: a | b, [id_2_root_cause_node[_] for _ in train_ids], set())
        test_rc_nodes = list(id_2_root_cause_node[_] for _ in test_ids)
        logger.info(
            f"{fault_type=!s:30} \n"
            f"train_length={len(train_ids):<5.0f} {train_ids=} \n"
            f"validation_length={len(validation_ids):<5.0f} {validation_ids=} \n"
            f"test_length={len(test_ids):<5.0f} {test_ids=} \n"
            f"({len(list(filter(lambda _: _ <= train_rc_nodes, test_rc_nodes))):<3.0f} recurring faults)")
        del train_ids, validation_ids, test_ids

    if balance_train_set:
        balanced_train_ids_list = []
        max_length = max([len(_) for _ in train_ids_list])
        for train_ids in train_ids_list:
            oversampling_ratio = max_length // len(train_ids)
            logger.info(f"repeat {train_ids} for {oversampling_ratio} times")
            balanced_train_ids_list.append(train_ids * oversampling_ratio)
        train_ids_list = balanced_train_ids_list
        del oversampling_ratio, max_length, train_ids, balanced_train_ids_list

    train_list = sum(train_ids_list, [])
    del train_ids_list

    sampled_train_list = []
    for i in train_list:
        if rng.random() <= train_set_sampling_ratio:
            sampled_train_list.append(i)
    # 除非train_set_sampling_ratio写成0，否则至少放一个训练数据
    if len(sampled_train_list) == 0 and train_set_sampling_ratio > 0:
        sampled_train_list.append(train_list[-1])
    train_list = sampled_train_list
    del sampled_train_list

    logger.info(
        f"{len(train_list)=} "
        f"{len(set(train_list))=} "
        f"{len(validation_list)=} "
        f"{len(test_list)=} "
    )
    return train_list, validation_list, test_list


def split_failures_by_recur(
        fault_df: pd.DataFrame, *, fdg: FDG = None, split: Tuple[float, float, float] = (0.3, 0.2, 0.5),
        train_set_sampling_ratio: float = 1.0, balance_train_set: bool = False, non_recur_index_train: int = -1, non_recur_index_test: int = -1
):
    rng = np.random.default_rng(233)  # the random seed should be fixed
    fault_type_2_id_list = defaultdict(list)
    id_2_root_cause_node = {}
    for idx, (_, fault) in enumerate(fault_df.iterrows()):
        # kpis = fault['kpi'] if not pd.isnull(fault['kpi']) else ""
        # 考虑根因KPI来对故障分类的话，有很多类别的故障数量太少（只有一两个）
        if fdg is not None:
            fault_type = tuple(fdg.instance_to_class(_) for _ in fault['root_cause_node'].split(";"))
        else:
            fault_type = tuple(fault['node_type'].split(";"))
        fault_type = list(fault_type)
        fault_type.sort()
        fault_type = tuple(fault_type)
        fault_type_2_id_list[fault_type].append(idx)
        id_2_root_cause_node[idx] = set(fault['root_cause_node'].split(";"))
    logger.info(
        f"fault ids with multiple root causes: "
        f"{[(k, len(v)) for k, v in id_2_root_cause_node.items() if len(v) > 1]}"
    )
    train_ids_list: List[List[int]] = []
    validation_list: List[int] = []
    test_list: List[int] = []
    num_fault_type = len(fault_type_2_id_list)
    non_recurring_list: List[int] = []
    non_recurring_list_test: List[int] = []
    non_recurring_fault_set: set = set()
    non_recurring_fault_sets = []
    for non_recurring_fault in list(fault_type_2_id_list.keys()):
        s = set(non_recurring_fault)
        if len(s)==1 and s not in non_recurring_fault_sets:
            non_recurring_fault_sets.append(s)
    if non_recur_index_train == -1:       # random select one as non-recurring failure
        non_recur_index_train = rng.integers(num_fault_type, size=1)[0]
        non_recurring_fault_set = set(list(fault_type_2_id_list.keys())[non_recur_index_train])
    else:
        non_recurring_fault_set = non_recurring_fault_sets[non_recur_index_train]
    if non_recur_index_test == -1:       # random select one as non-recurring failure to test
        non_recur_index_test = rng.integers(num_fault_type, size=1)[0]
        while non_recur_index_test == non_recur_index_train:
            non_recur_index_test = rng.integers(num_fault_type, size=1)[0]
        non_recurring_fault_set_test = set(list(fault_type_2_id_list.keys())[non_recur_index_test])
    else:
        non_recurring_fault_set_test = non_recurring_fault_sets[non_recur_index_test]
    for fault_type, ids in fault_type_2_id_list.items():
        if non_recurring_fault_set <= set(fault_type):
            _train_split = 0
            _valid_split = 0
            non_recurring_list.extend(ids)
        elif non_recurring_fault_set_test <= set(fault_type):
            _train_split = 0
            _valid_split = 0
            non_recurring_list_test.extend(ids)
        else:
            _train_split = max(int(len(ids) * split[0]), 1)
            _valid_split = max(int(len(ids) * (split[0] + split[1])), 1)
        rng.shuffle(ids)

        train_ids = ids[0:_train_split]
        train_ids_list.append(train_ids)

        validation_ids = ids[_train_split:_valid_split]
        validation_list.extend(validation_ids)

        test_ids = ids[_valid_split:]
        test_list.extend(test_ids)

        del _train_split, _valid_split

        train_rc_nodes = reduce(lambda a, b: a | b, [id_2_root_cause_node[_] for _ in train_ids], set())
        test_rc_nodes = list(id_2_root_cause_node[_] for _ in test_ids)
        logger.info(
            f"{fault_type=!s:30} \n"
            f"train_length={len(train_ids):<5.0f} {train_ids=} \n"
            f"validation_length={len(validation_ids):<5.0f} {validation_ids=} \n"
            f"test_length={len(test_ids):<5.0f} {test_ids=} \n"
            f"({len(list(filter(lambda _: _ <= train_rc_nodes, test_rc_nodes))):<3.0f} recurring faults)")
        del train_ids, validation_ids, test_ids

    if balance_train_set:
        balanced_train_ids_list = []
        max_length = max([len(_) for _ in train_ids_list])
        for train_ids in train_ids_list:
            if len(train_ids) == 0:
                continue
            oversampling_ratio = max_length // len(train_ids)
            logger.info(f"repeat {train_ids} for {oversampling_ratio} times")
            balanced_train_ids_list.append(train_ids * oversampling_ratio)
        train_ids_list = balanced_train_ids_list
        del oversampling_ratio, max_length, train_ids, balanced_train_ids_list

    train_list = sum(train_ids_list, [])
    del train_ids_list

    sampled_train_list = []
    for i in train_list:
        if rng.random() <= train_set_sampling_ratio:
            sampled_train_list.append(i)
    # 除非train_set_sampling_ratio写成0，否则至少放一个训练数据
    if len(sampled_train_list) == 0 and train_set_sampling_ratio > 0:
        sampled_train_list.append(train_list[-1])
    train_list = sampled_train_list
    del sampled_train_list

    logger.info(
        f"{len(train_list)=} "
        f"{len(set(train_list))=} "
        f"{len(validation_list)=} "
        f"{len(test_list)=} \n"
        f"{non_recurring_fault_set=!s:30} "
        f"{non_recurring_list=}"
        f"{non_recurring_fault_set_test=!s:30} "
        f"{non_recurring_list_test=}"
    )
    return train_list, validation_list, test_list, non_recurring_list, non_recurring_fault_set, non_recurring_list_test, non_recurring_fault_set_test


def get_non_recur_lists(
        fault_df: pd.DataFrame, *, fdg: FDG = None
) -> Tuple[List[int], List[int], List[int]]:
    fault_type_2_id_list = defaultdict(list)
    id_2_root_cause_node = {}
    for idx, (_, fault) in enumerate(fault_df.iterrows()):
        print(f"id: {idx}, fault: {fault}")
        if fdg is not None:
            fault_type = tuple(fdg.instance_to_class(_) for _ in fault['root_cause_node'].split(";"))
        else:
            fault_type = tuple(fault['node_type'].split(";"))
        fault_type = list(fault_type)
        fault_type.sort()
        fault_type = tuple(fault_type)
        fault_type_2_id_list[fault_type].append(idx)
        id_2_root_cause_node[idx] = set(fault['root_cause_node'].split(";"))
    non_recurring_fault_sets = []
    for non_recurring_fault in list(fault_type_2_id_list.keys()):
        s = set(non_recurring_fault)
        if len(s)==1 and s not in non_recurring_fault_sets:
            non_recurring_fault_sets.append(s)
    return non_recurring_fault_sets


def split_failures_by_drift(
        fault_df: pd.DataFrame, *, fdg: FDG = None, split: Tuple[float, float, float] = (0.8, 0.2, 0.0), drift_time: int,
        train_set_sampling_ratio: float = 1.0, balance_train_set: bool = False
) -> Tuple[List[int], List[int], List[int]]:
    random_seed = np.random.randint(1,1000)
    rng = np.random.default_rng(random_seed)  # the random seed should be fixed
    fault_type_2_id_list = defaultdict(list)
    id_2_root_cause_node = {}
    test_list: List[int] = []
    drift_list: List[int] = []
    for idx, (_, fault) in enumerate(fault_df.iterrows()):
        if fault['timestamp'] >= int(drift_time):
            test_list.append(idx)
            drift_list.append(idx)
            continue
        if fdg is not None:
            fault_type = tuple(fdg.instance_to_class(_) for _ in fault['root_cause_node'].split(";"))
        else:
            fault_type = tuple(fault['node_type'].split(";"))
        fault_type = list(fault_type)
        fault_type.sort()
        fault_type = tuple(fault_type)
        fault_type_2_id_list[fault_type].append(idx)
        id_2_root_cause_node[idx] = set(fault['root_cause_node'].split(";"))
    logger.info(
        f"fault ids with multiple root causes: "
        f"{[(k, len(v)) for k, v in id_2_root_cause_node.items() if len(v) > 1]}"
    )
    train_ids_list: List[List[int]] = []
    validation_list: List[int] = []
    for fault_type, ids in fault_type_2_id_list.items():
        _train_split = max(int(len(ids) * split[0]), 1)
        _valid_split = max(int(len(ids) * (split[0] + split[1])), 1)
        rng.shuffle(ids)

        train_ids = ids[0:_train_split]
        train_ids_list.append(train_ids)

        validation_ids = ids[_train_split:_valid_split]
        validation_list.extend(validation_ids)

        test_ids = ids[_valid_split:]
        test_list.extend(test_ids)

        del _train_split, _valid_split

        train_rc_nodes = reduce(lambda a, b: a | b, [id_2_root_cause_node[_] for _ in train_ids], set())
        test_rc_nodes = list(id_2_root_cause_node[_] for _ in test_ids)
        logger.info(
            f"{fault_type=!s:30} \n"
            f"train_length={len(train_ids):<5.0f} {train_ids=} \n"
            f"validation_length={len(validation_ids):<5.0f} {validation_ids=} \n"
            f"test_length={len(test_ids):<5.0f} {test_ids=} \n"
            f"({len(list(filter(lambda _: _ <= train_rc_nodes, test_rc_nodes))):<3.0f} recurring faults)")
        del train_ids, validation_ids, test_ids

    if balance_train_set:
        balanced_train_ids_list = []
        max_length = max([len(_) for _ in train_ids_list])
        for train_ids in train_ids_list:
            oversampling_ratio = max_length // len(train_ids)
            logger.info(f"repeat {train_ids} for {oversampling_ratio} times")
            balanced_train_ids_list.append(train_ids * oversampling_ratio)
        train_ids_list = balanced_train_ids_list
        del oversampling_ratio, max_length, train_ids, balanced_train_ids_list

    train_list = sum(train_ids_list, [])
    del train_ids_list

    sampled_train_list = []
    for i in train_list:
        if rng.random() <= train_set_sampling_ratio:
            sampled_train_list.append(i)
    # 除非train_set_sampling_ratio写成0，否则至少放一个训练数据
    if len(sampled_train_list) == 0 and train_set_sampling_ratio > 0:
        sampled_train_list.append(train_list[-1])
    train_list = sampled_train_list
    del sampled_train_list

    num_train, num_valid, num_test, num_drift, num_non_drift = len(train_list), len(validation_list), len(test_list), len(drift_list), len(test_list) - len(drift_list)
    failure_num = len(train_list) + len(validation_list) + len(test_list)
    logger.info(
        f"All {failure_num} failures: "
        f"{num_train=} "
        f"{len(set(train_list))=} "
        f"{num_valid=} "
        f"{num_test=} "
    )
    logger.info(
        f"Train: Valid: Test = {num_train/failure_num:.2f}: {num_valid/failure_num:.2f}: {num_test/failure_num:.2f}"
        f"In Test Set, drift: non-drift = {num_drift/num_test:.2f}: {num_non_drift/num_test:.2f}"
    )
    return train_list, validation_list, test_list, drift_list