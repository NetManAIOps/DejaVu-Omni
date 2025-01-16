import pickle
from collections import defaultdict
from functools import lru_cache, reduce, cached_property
from itertools import groupby
from pathlib import Path
from pprint import pformat
from typing import Tuple, Dict, List, Set, Union, Callable, Optional, Iterable

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch as th
from dgl import DGLHeteroGraph, heterograph
from loguru import logger
from networkx import DiGraph
from numpy.lib.stride_tricks import sliding_window_view
from pyprof import profile

__all__ = [
    'FDG'
]

from failure_dependency_graph.generate_virtual_FDG import generate_virtual_FDG
from failure_dependency_graph.parse_yaml_graph_config import parse_yaml_graph_config
from utils import read_dataframe, py_object_to_nb_object


class FDG:
    """
    A container for FDG, metrics and failures
    """
    _nx_failure_graphs: List[DiGraph]
    _nx_overall_graph: DiGraph

    @profile
    def __init__(
            self, *, metrics: pd.DataFrame, failures: pd.DataFrame,
            anomaly_direction_constraint: Dict[str, str] = None,
            graph: Optional[nx.DiGraph] = None, failure_graphs: Optional[List[nx.DiGraph]] = None,
    ):
        """
        :param graph: a networkx graph defines the failure dependency graph structure
            A node should have a type (str) and metrics (a list of str)
            A edge should have a type (str)
        :param metrics:
            The columns should be name, timestamp (in seconds) and value
        """
        assert set(metrics.columns) >= {'timestamp', 'value', "name"}, f"{metrics.columns=}"
        assert set(failures.columns) >= {'timestamp', 'root_cause_node'}, f"{failures.columns=}"
        assert graph is not None or failure_graphs is not None, "Either graph or failure_graphs should be provided"
        assert failure_graphs is None or len(failure_graphs) == len(failures), \
            f"The length of {len(failure_graphs)=} should be equal to the length of {len(failures)=}"
        if failure_graphs is None:
            # noinspection PyTypeChecker
            failure_graphs: List[DiGraph] = [graph.copy() for _ in range(len(failures))]
        if graph is None:
            graph = nx.compose_all(failure_graphs)
        self._nx_overall_graph = graph
        self._nx_failure_graphs = failure_graphs

        self._node_list: Dict[str, List[str]] = dict(map(
            lambda pair: (pair[0], [_[0] for _ in pair[1]]),
            groupby(sorted(
                graph.nodes(data=True), key=lambda _: _[1]['type']
            ), key=lambda _: _[1]['type'])
        ))

        self._node_to_idx: Dict[str, Dict[str, int]] = {
            node_type: {node: idx for idx, node in enumerate(node_list)}
            for node_type, node_list in self._node_list.items()
        }
        self._node_to_class: Dict[str, str] = dict(sum(
            [[(node, fc) for node in nodes] for fc, nodes in self._node_list.items()],
            []
        ))

        self._all_metrics_set: Set[str] = reduce(
            lambda a, b: a | b,
            [set(data['metrics']) for _, data in graph.nodes(data=True)],
            set()
        )
        self._metric_size_dict: Dict[str, int] = self.__get_metric_size_dict(graph)
        # 假设每个类型的指标都是相同的，传入的顺序也是相同的
        self._node_metrics_dict: Dict[str, List[str]] = self.__get_node_metrics_dict(graph)
        self._node_type_metrics_dict: Dict[str, List[str]] = self.__get_node_type_metrics_dict(
            self._node_metrics_dict, self._node_list
        )

        self._metrics_df = metrics[metrics.name.isin(self._all_metrics_set)]
        self._overall_hg = self.__get_hg(graph)
        logger.debug(f"All edge types: {pformat(self._overall_hg.etypes)}")
        self._node_types = self._overall_hg.ntypes
        assert self.overall_hg.ntypes == self.failure_classes

        self._faults_df = failures

        logger.info(f"all {self.n_components} components: {self.components}")
        logger.info(f"the number of nodes: {self.n_failure_instances}")
        logger.info(f"all ({len(self.failure_classes)}) node types: {self.failure_classes}")
        logger.info(f"the number of metrics: "
                    f"{sum(self.metric_number_dict.values())} (metric_type) "
                    f"{sum([len(_) for _ in self.FI_metrics_dict.values()])} (name)")

        logger.info(f"The metrics of each node type: \n{pformat(self.FC_metrics_dict)}")

        self._anomaly_direction_constraint = {_.split('##')[1]: 'b' for _ in self._all_metrics_set}
        self._anomaly_direction_constraint.update(
            anomaly_direction_constraint if anomaly_direction_constraint is not None else {}
        )

    @staticmethod
    @profile
    def load(
            dir_path: Union[Path, str, None] = None, *,
            graph_path: Union[Path, str, None] = None,
            metrics_path: Union[Path, str, None] = None,
            faults_path: Union[Path, str, None] = None,
            use_anomaly_direction_constraint: bool = False,
            return_real_path: bool = False,
            loaded_FDG: Optional['FDG'] = None,
            flush_dataset_cache: bool = False,
            non_recurring_fault_set: Optional[set] = None
    ) -> Union['FDG', Tuple['FDG', Dict]]:
        """
        :param flush_dataset_cache: If False, try to load the FDG from pickle in `dir_path`
        :param dir_path: read "graph.yaml", "metrics.norm.pkl", and "faults.csv" in this directory by default
        :param graph_path: overwrite the default graph path
        :param metrics_path: overwrite the default metrics path
        :param faults_path: overwrite the default faults path
        :param use_anomaly_direction_constraint: whether to use anomaly direction constraint
        :param return_real_path: whether to return the real path of the loaded files
        :param loaded_FDG: if not None, use the given FDG instead of loading from files
        :param non_recurring_fault_set: if not None, delete recurring failure types in FDG
        :return: a FDG object,
            (optional) and a dict of the real paths of the loaded files whose keys are 'graph', 'metrics', and 'faults'
        """
        if dir_path is not None:
            dir_path = Path(dir_path)

            if str(dir_path).startswith("/dev/generate_random_FDG"):
                fdg = generate_virtual_FDG(
                    **{
                        _.split("=")[0]: float(_.split("=")[1])
                        for _ in str(dir_path.relative_to('/dev/generate_random_FDG')).split("/")
                    }
                )
                if return_real_path:
                    return fdg, {
                        'graph': "/dev/generate_random_FDG/graph",
                        'metrics': "/dev/generate_random_FDG/metrics",
                        "faults": "/dev/generate_random_FDG/faults"}
                else:
                    return fdg

            assert dir_path.is_dir() and dir_path.exists(), dir_path
        graph_path = dir_path / 'graph.yml' if graph_path is None else Path(graph_path)
        # metrics_path = dir_path / "metrics.pkl" if metrics_path is None else Path(metrics_path)
        metrics_path = dir_path / "metrics.norm.pkl" if metrics_path is None else Path(metrics_path)
        faults_path = dir_path / "faults.csv" if faults_path is None else Path(faults_path)
        if not graph_path.exists():
            graph_path = None
        assert metrics_path.exists(), metrics_path
        assert faults_path.exists(), faults_path

        if not loaded_FDG or non_recurring_fault_set != None:
            fdg = FDG._load_FDG(
                dir_path=dir_path,
                graph_path=graph_path,
                metrics_path=metrics_path,
                faults_path=faults_path,
                use_anomaly_direction_constraint=use_anomaly_direction_constraint,
                flush_dataset_cache=flush_dataset_cache,
                non_recurring_fault_set=non_recurring_fault_set,
            )
        else:
            fdg = loaded_FDG
        if return_real_path:
            return fdg, {'graph': graph_path, 'metrics': metrics_path, "faults": faults_path}
        else:
            return fdg

    @staticmethod
    def _load_FDG(
            dir_path: Path, graph_path: Optional[Path], faults_path: Path, metrics_path: Path,
            use_anomaly_direction_constraint: bool, flush_dataset_cache=False, non_recurring_fault_set: Optional[set]=None
    ):
        if not use_anomaly_direction_constraint:
            anomaly_direction_constraint = {}
        else:
            try:
                with open(dir_path / "anomaly_direction_constraint.json", 'r') as f:
                    import json
                    anomaly_direction_constraint = json.load(f)
            except Exception as e:
                logger.error(e)
                anomaly_direction_constraint = {}
        failures_df = read_dataframe(faults_path)
        failures_graph_paths = [
            _ for _ in
            [dir_path / 'graphs' / f"graph_{t:.0f}.yml" for t in failures_df["timestamp"]]
            if _.exists()
        ]
        latest_source_modification_time = max(
            [
                _.stat().st_mtime
                for _ in failures_graph_paths + [graph_path, faults_path, metrics_path]
                if _ is not None and _.exists()
            ]
        )
        pickled_FDG_path: Path = dir_path / "FDG.pkl"
        is_loaded_from_pickle = False
        try:
            if pickled_FDG_path.exists() \
                    and pickled_FDG_path.stat().st_mtime >= latest_source_modification_time \
                    and not flush_dataset_cache:
                logger.info(f"Loading FDG from {pickled_FDG_path}")
                with open(pickled_FDG_path, 'rb') as f:
                    fdg = pickle.load(f)
                    is_loaded_from_pickle = True
        except Exception as e:
            logger.exception("Read pickled FDG error", exception=e)
        if not is_loaded_from_pickle:
            if len(failures_graph_paths) == len(failures_df):
                failure_graphs = [FDG.modify_graph_by_node_type(parse_yaml_graph_config(_), non_recurring_fault_set) for _ in failures_graph_paths]
            else:
                failure_graphs = None
            logger.debug(f"Load CDP: {graph_path=} {metrics_path=} {faults_path=} {failures_graph_paths=}")
            fdg = FDG(
                graph=FDG.modify_graph_by_node_type(parse_yaml_graph_config(graph_path), non_recurring_fault_set) if graph_path is not None else None,
                failure_graphs=failure_graphs,
                metrics=read_dataframe(metrics_path),
                failures=failures_df,
                anomaly_direction_constraint=anomaly_direction_constraint,
            )
            with open(pickled_FDG_path, 'wb') as f:
                pickle.dump(fdg, f, protocol=-1)
        else:
            fdg = fdg  # trick IDE
        return fdg
    
    @staticmethod
    def modify_graph_by_node_type(g: nx.DiGraph, node_types: Optional[set] = None):
        '''
        delete nodes and edges from g where node type in `node_types`.
        '''
        if node_types is None:
            return g
        nodes = [node_index for node_index, node_attr in g.nodes(data=True) if node_attr['type'] in node_types]
        g.remove_nodes_from(nodes)
        return g

    #######################
    # Metrics
    #######################
    @property
    def metrics_df(self) -> pd.DataFrame:
        """
        :return: A dataframe contains the following columns: timestamp, name, value
        """
        return self._metrics_df

    @property
    def metric_mean_dict(self) -> Dict[str, float]:
        """
        :return: The average value of each metric
        """
        return self._metrics_df.groupby('name')['value'].mean().to_dict()

    @cached_property
    def metric_mean_dict_nb(self):
        return py_object_to_nb_object(self.metric_mean_dict)

    @cached_property
    def metric_max_dict(self) -> Dict[str, float]:
        ret = {m: 0 for m in self.metrics}
        ret.update(self._metrics_df.groupby('name')['value'].max().to_dict())
        return ret

    @cached_property
    def metric_min_dict(self) -> Dict[str, float]:
        ret = {m: 0 for m in self.metrics}
        ret.update(self._metrics_df.groupby('name')['value'].min().to_dict())
        return ret

    @cached_property
    def metric_kind_max_dict(self) -> Dict[str, float]:
        return {
            metric_kind: max([self.metric_max_dict[m] for m in metrics])
            for metric_kind, metrics in self.metric_kinds_to_metrics_dict.items()
        }

    @cached_property
    def metric_kind_min_dict(self) -> Dict[str, float]:
        return {
            metric_kind: min([self.metric_min_dict[m] for m in metrics])
            for metric_kind, metrics in self.metric_kinds_to_metrics_dict.items()
        }

    @property
    def metric_number_dict(self) -> Dict[str, int]:
        """
        :return: The number of metrics of each node type
        """
        return self._metric_size_dict

    @property
    def FI_metrics_dict(self) -> Dict[str, List[str]]:
        """
        :return: The list of metrics of each failure instance
        """
        return self._node_metrics_dict

    @cached_property
    def FI_metrics_dict_nb(self):
        return py_object_to_nb_object(self.FI_metrics_dict)
    
    @cached_property
    def FI_component_metrics_dict(self) -> Dict[str, List[str]]:
        """
        :return: The list of component metrics of each failure instance
        """
        ret: Dict[str, List[str]] = {}
        for key, value in self.failure_instance_to_component_dict.items():
            ret[key] = self.component_to_metrics_dict[value]
        return ret
    
    @cached_property
    def FI_component_metrics_dict_nb(self):
        return py_object_to_nb_object(self.FI_component_metrics_dict)

    @property
    def FC_metrics_dict(self) -> Dict[str, List[str]]:
        """
        :return: The list of metrics of each failure class
        """
        return self._node_type_metrics_dict

    @cached_property
    def metric_kinds_to_metrics_dict(self) -> Dict[str, Set[str]]:
        """
        :return: The list of metrics of each metric kind
        """
        ret = defaultdict(set)
        for metric in self.metrics:
            ret[metric.split("##")[1]].add(metric)
        return dict(ret)

    @cached_property
    def metric_kinds(self) -> Set[str]:
        return set(self.metric_kinds_to_metrics_dict.keys())

    @cached_property
    def metrics(self) -> List[str]:
        return list(self.metric_to_FI_list_dict.keys())

    @cached_property
    def metric_to_FI_list_dict(self) -> Dict[str, List[str]]:
        """
        :return: The mapping from metrics to failure instances
        """
        ret = defaultdict(list)
        for FI, metrics in self.FI_metrics_dict.items():
            for metric in metrics:
                ret[metric].append(FI)
        return dict(ret)

    @property
    def anomaly_direction_constraint(self) -> Dict[str, str]:
        """
        :return:
            'u', upside the baseline is abnormal
            'd', downside the baseline is abnormal
            'b', both sides are abnormal
        """
        return self._anomaly_direction_constraint

    #######################
    # Components
    #######################
    @cached_property
    def components(self) -> List[str]:
        """
        :return: The set of components
        """
        return sorted(list({_.split("##")[0] for _ in self.metrics}))
    
    @property
    def n_components(self) -> int:
        """
        :return: The number of components
        """
        return len(self.components)
    
    @cached_property
    def component_to_component_id(self) -> Dict[str, int]:
        tmp: Dict[str, int] = {
            component: idx
            for idx, component in enumerate(self.components)
        }
        return tmp
    
    @cached_property
    def component_fi_index_project_list(self) -> List[int]:
        tmp = []
        for component_class in self.component_classes:
            for component in self.component_class_to_components_dict[component_class]:
                for fi in self.component_to_failure_instances_dict[component]:
                    tmp.append(self.instance_to_gid(fi))
        ret = np.empty(len(tmp), dtype=int)
        for i, index in enumerate(tmp):
            ret[index] = i
        return list(ret)

    @lru_cache(maxsize=None)
    def get_component_neighbor(self, component: str) -> List[str]:
        """
        :param component: The component
        :return: The list of neighbor components
        """
        assert component in self.components, f"{component=} is not in {self.components=}"
        failure_instances = self.component_to_failure_instances_dict[component]
        neighbor_failure_instances = set.union(*[
            set(self.get_failure_instance_neighbors(fi)) for fi in failure_instances
        ])
        neighbor_component_node_types_dist = {
                                  self.failure_instance_to_component_dict[fi] for fi in neighbor_failure_instances
                              } - {component}
        return sorted(list(neighbor_component_node_types_dist))

    @cached_property
    def failure_instance_to_component_dict(self) -> Dict[str, str]:
        """
        :return: The mapping from failure instance to component
        """
        ret: Dict[str, str] = {}
        for fi, metrics in self.FI_metrics_dict.items():
            failure_instances = list({_.split("##")[0] for _ in metrics})
            assert len(failure_instances) == 1, \
                f"The failure instance {fi} has more than one component. It contains {metrics}"
            ret[fi] = failure_instances[0]
        return ret

    @cached_property
    def component_to_failure_instances_dict(self) -> Dict[str, List[str]]:
        """
        key: Each component
        value: The list of failure instances belonging to the component
        :return:
        """
        ret = defaultdict(set)
        for fi, component in self.failure_instance_to_component_dict.items():
            ret[component].add(fi)
        return {
            k: sorted(list(v)) for k, v in ret.items()
        }
    
    @cached_property
    def fi_to_component_class_dict(self) -> Dict[str, str]:
        """
        :return: The mapping from failure instance to component class
        """
        ret: Dict[str, str] = {}
        for fi, metrics in self.FI_metrics_dict.items():
            failure_instances = list({_.split("##")[0] for _ in metrics})
            assert len(failure_instances) == 1, \
                f"The failure instance {fi} has more than one component. It contains {metrics}"
            ret[fi] = failure_instances[0].split('_')[0]
        return ret

    @cached_property
    def component_class_to_fi_dict(self) -> Dict[str, List[str]]:
        """
        key: Each component class
        value: The list of failure instances belonging to the component class
        :return:
        """
        ret = defaultdict(set)
        for fi, component in self.fi_to_component_class_dict.items():
            ret[component].add(fi)
        return {
            k: sorted(list(v)) for k, v in ret.items()
        }
    
    @cached_property
    def component_class_to_components_dict(self) -> Dict[str, List[str]]:
        """
        key: Each component class
        value: The list of components belonging to the component class
        :return:
        """
        component_classes = defaultdict(list)
        for component in self.components:
            metric_kinds = frozenset({_.split("##")[1] for _ in self.component_to_metrics_dict[component]})
            # if metric_kinds == frozenset({'fake'}):
            #     continue
            component_classes[metric_kinds].append(component)
        ret = dict()
        for k, v in component_classes.items():
            components = sorted(v)
            ret[components[0]+'_like'] = components
        return ret
    
    @cached_property
    def component_to_component_class_dict(self) -> Dict[str, List[str]]:
        ret = {}
        for component_class, components in self.component_class_to_components_dict.items():
            for component in components:
                ret[component] = component_class
        return ret

    @cached_property
    def component_class_to_components_dict_nb(self):
        return py_object_to_nb_object(self.component_class_to_components_dict)

    @cached_property
    def component_to_failure_instances_nb(self):
        return py_object_to_nb_object(self.component_to_failure_instances_dict)

    @cached_property
    def component_class_to_fi_size_dict(self) -> Dict[str, int]:
        """
        key: Each component class
        value: The number of failure instances belonging to the component class
        :return:
        """
        return {
            k: len(self.component_to_failure_instances_dict[v[0]]) for k, v in self.component_class_to_components_dict.items()
        }
    
    @cached_property
    def component_to_metrics_dict(self) -> Dict[str, List[str]]:
        """
        :return: The mapping from components to metrics
        """
        ret = dict()
        for component in self.components:
            fi_list = self.component_to_failure_instances_dict[component]
            ret[component] = sorted(list(set.union(*[
                set(self.FI_metrics_dict[fi]) for fi in fi_list
            ])))
        return ret
    
    @cached_property
    def component_to_metrics_dict_nb(self):
        return py_object_to_nb_object(self.component_to_metrics_dict)
    
    @cached_property
    def component_class_metrics_number_dict(self) -> Dict[str, List[str]]:
        """
        :return: The number of metrics of each component class
        """
        ret = dict()
        for component_class in self.component_classes:
            components = self.component_class_to_components_dict[component_class]
            ret[component_class] = len(self.component_to_metrics_dict[components[0]])
        return ret

    @lru_cache(maxsize=None)
    def get_failure_instance_from_component_and_class(self, component: str, failure_class: str) -> str:
        _ret = set(self.component_to_failure_instances_dict[component]) & set(self.failure_instances[failure_class])
        assert len(_ret) == 1, \
            f"{set(self.component_to_failure_instances_dict[component])=} " \
            f"{set(self.failure_instances[failure_class])}"
        return _ret.pop()

    @cached_property
    def component_in_classes(self) -> List[List[str]]:
        component_classes = defaultdict(list)
        for component in self.components:
            metric_kinds = frozenset({_.split("##")[1] for _ in self.component_to_metrics_dict[component]})
            # if metric_kinds == frozenset({'fake'}):
            #     continue
            component_classes[metric_kinds].append(component)
        return [sorted(v) for k, v in component_classes.items()]

    @cached_property
    def component_to_class_dict(self) -> Dict[str, List[str]]:
        ret = {}
        for component_class in self.component_in_classes:
            for component in component_class:
                ret[component] = component_class
        return ret

    @cached_property
    def component_classes(self) -> List[str]:
        """
        :return: The components classes
        """
        return sorted([components[0]+'_like' for components in self.component_in_classes])

    #######################
    # Nodes
    #######################
    @property
    def n_failure_instances(self) -> int:
        """
        :return: The number of failure instances
        """
        return self.overall_hg.number_of_nodes()

    @property
    def failure_instances(self) -> Dict[str, List[str]]:
        """
        :return: A dict mapping a failure class to its instances
        """
        return self._node_list

    def get_failure_instance_neighbors(self, failure_instance: str) -> List[str]:
        """
        :param failure_instance: The failure instance
        :return: The list of neighbors of the failure instance
        """
        return sorted(list(self._nx_overall_graph.neighbors(failure_instance)))

    @cached_property
    def failure_instances_nb(self):
        return py_object_to_nb_object(self.failure_instances)

    @cached_property
    def flatten_failure_instances(self) -> List[str]:
        """
        :return: A list of failure instance names, where the indices are the gids of the instances
        """
        return sum([self.failure_instances[_] for _ in self.failure_classes], [])

    @property
    def failure_classes(self) -> List[str]:
        return self._node_types
    
    @property
    def failure_class_to_id(self) -> Dict[str, int]:
        tmp: Dict[str, int] = {
            fc: idx
            for idx, fc in enumerate(self.failure_classes)
        }
        return tmp

    @cached_property
    def invalid_failure_class_indices(self) -> List[int]:
        """
        从来不作为根因的failure class的indices
        :return:
        """
        all_failure_classes_set = set(self.failure_classes)
        rc_failure_classes = set.union(*[
            {self.instance_to_class(_) for _ in self.root_cause_instances_of(fid)}
            for fid in self.failure_ids
        ])
        return [self.failure_classes.index(_) for _ in all_failure_classes_set - rc_failure_classes]

    def instance_to_class(self, name: str) -> str:
        """
        Map a failure instance to its failure class
        :param name:
        :return:
        """
        return self._node_to_class[name]

    def gid_to_instance(self, gid: int) -> str:
        """
        Map a global id to a failure instance name
        :param gid: the global id of a failure instance
        :return: the name of the instance
        """
        node_type, node_typed_id = self.gid_to_local_id(gid)
        return self.failure_instances[node_type][node_typed_id]

    def gid_to_component(self, gid: int) -> str:
        """
        Map a global id to a component name
        :param gid: the global id of a component
        :return: the name of the component
        """
        fi = self.gid_to_instance(gid)
        return self.failure_instance_to_component_dict[fi]
    
    def gid_to_failure_class(self, gid: int) -> str:
        """
        Map a global id to a failure_class name
        :param gid: the global id of a failure_class
        :return: the name of the failure_class
        """
        fi = self.gid_to_instance(gid)
        return self.instance_to_class(fi)
    
    def gid_to_fc_id(self, gid: int) -> str:
        """
        Map a global id to a failure_class id
        :param gid: the global id of a failure_class
        :return: the id of the failure_class
        """
        fc = self.gid_to_failure_class(gid)
        return self.failure_class_to_id[fc]
    
    def gid_to_component_id(self, gid: int) -> str:
        """
        Map a global id to a component id
        :param gid: the global id of a component
        :return: the id of the component
        """
        component = self.gid_to_component(gid)
        return self.component_to_component_id[component]

    def instance_to_gid(self, name: str) -> int:
        """
        Map a failure instance name to a global id
        :param name:
        :return:
        """
        _node_type = self.instance_to_class(name)
        _local_index = self.failure_instances[_node_type].index(name)
        return self.local_id_to_gid(_node_type, _local_index)

    def gid_to_local_id(self, global_id: int) -> Tuple[str, int]:
        """
        Calculate the failure class and local id of a global id
        :param global_id:
        :return: (failure class, local id). Local id is the index of the failure instance in the failure class
        """
        resolver = _get_global_id_resolver(self.overall_hg)
        node_type, node_typed_id = resolver(global_id)
        return node_type, node_typed_id

    def local_id_to_gid(self, failure_class: str, local_id: int) -> int:
        """
        Calculate the global id of a failure class and local id
        :param failure_class:
        :param local_id: the local id of a failure instance inside the failure class
        :return:
        """
        getter = _get_global_id_getter(self.overall_hg)
        return getter(failure_class, local_id)

    def local_id_to_instance(self, failure_class: str, local_id: int) -> str:
        return self.gid_to_instance(self.local_id_to_gid(failure_class, local_id))

    def instance_to_local_id(self, name: str) -> Tuple[str, int]:
        return self.gid_to_local_id(self.instance_to_gid(name))

    #######################
    # DGLGraph
    #######################

    @property
    def overall_hg(self) -> DGLHeteroGraph:
        return self._overall_hg

    @cached_property
    def overall_homo(self) -> dgl.DGLGraph:
        return self.convert_to_homo(self.overall_hg)

    def overall_networkx(self) -> nx.DiGraph:
        return self._nx_overall_graph

    def networkx_graph_at(self, fid: int) -> nx.DiGraph:
        return self._nx_failure_graphs[fid]

    @lru_cache(maxsize=None)
    def hetero_graph_at(self, failure_id: int) -> DGLHeteroGraph:
        return self.__get_hg(self._nx_failure_graphs[failure_id])

    @lru_cache(maxsize=None)
    def homo_graph_at(self, failure_id: int) -> dgl.DGLGraph:
        return self.convert_to_homo(self.hetero_graph_at(failure_id))

    def convert_to_homo(self, hg: DGLHeteroGraph) -> dgl.DGLGraph:
        _gid_dict = {}
        _og: dgl.DGLGraph = dgl.to_homogeneous(hg)
        for _node, _ori_id, _ori_type in zip(_og.nodes(), _og.ndata['_ID'], _og.ndata['_TYPE']):
            _gid_dict[int(_node)] = self.local_id_to_gid(hg.ntypes[_ori_type], _ori_id)
        return dgl.graph(
            data=([_gid_dict[int(_)] for _ in _og.edges()[0]], [_gid_dict[int(_)] for _ in _og.edges()[1]]),
            num_nodes=self.overall_hg.number_of_nodes(),
            device=hg.device,
            idtype=hg.idtype,
        )

    #######################
    # Failures
    #######################
    @property
    def failures_df(self) -> pd.DataFrame:
        return self._faults_df

    def failure_at(self, fid: int):
        return self._faults_df.iloc[fid]

    def root_cause_instances_of(self, fid: int) -> List[str]:
        return self.failure_at(fid)['root_cause_node'].split(';')

    @property
    def failure_ids(self) -> List[int]:
        return list(range(len(self.failures_df)))

    @cached_property
    def timestamp_range(self) -> Tuple[int, int]:
        fault_ts_min = self.failure_timestamps()[0]
        fault_ts_max = self.failure_timestamps()[-1]
        valid_ts_min = self.valid_timestamps[0]
        valid_ts_max = self.valid_timestamps[-1]
        return min(valid_ts_min, fault_ts_min), max(valid_ts_max, fault_ts_max)

    @cached_property
    def valid_timestamps(self) -> np.ndarray:
        """
        至少一个指标是有值的时间戳
        :return:
        """
        return np.sort(self.metrics_df['timestamp'].unique())

    def failure_timestamps(self, duration=5, granularity=60, before_duration=0) -> np.ndarray:
        start = self.failures_df['timestamp'].unique().reshape(-1, 1)
        expand = (np.arange(-before_duration, duration + 1) * granularity).reshape(1, -1)
        return np.sort(np.unique((start + expand).reshape(-1)))

    def failure_time_windows(
            self, duration=5, granularity=60, before_duration=0, window_size: int = 10,
            failure_ids: Optional[Iterable[int]] = None
    ) -> np.ndarray:
        """
        :param failure_ids:
        :param duration:
        :param granularity:
        :param before_duration:
        :param window_size:
        :return: A ndarray of timestamps, which is of shape (num_of_windows, window_size)
        """
        assert duration + before_duration >= window_size, \
            f"{duration=} + {before_duration=} should be larger than {window_size=}"
        ret = []
        failure_ids = self.failure_ids if failure_ids is None else list(failure_ids)
        for fault_ts in [self.failure_at(fid)['timestamp'] for fid in failure_ids]:
            ret.append(sliding_window_view(
                np.arange(fault_ts - before_duration * granularity, fault_ts + duration * granularity + 1, granularity),
                window_shape=window_size, axis=-1,
            ))
        return np.concatenate(ret, axis=0)

    def normal_timestamps(self, granularity=60, duration=10, before_duration=0) -> np.ndarray:
        return np.sort(np.asarray(
            list(set(
                self.valid_timestamps[duration:-duration] if duration > 0 else self.valid_timestamps
            ) - set(
                self.failure_timestamps(
                    duration=duration, granularity=granularity, before_duration=before_duration
                )
            ))
        ))

    def normal_time_windows(
            self, granularity=60, duration=10, before_duration=0,
            window_size: int = 10
    ) -> np.ndarray:
        """
        :param window_size:
        :param granularity:
        :param duration:
        :param before_duration:
        :return:
        """
        timestamps = self.normal_timestamps(granularity=granularity, duration=duration, before_duration=before_duration)
        split_indices = np.where(np.diff(timestamps) > granularity)[0] + 1
        ret = []
        for segments in np.split(timestamps, split_indices, axis=-1):
            if len(segments) < window_size:
                continue
            ret.append(sliding_window_view(segments, window_shape=window_size, axis=-1))
        return np.concatenate(ret, axis=0)

    ###########################
    # Init
    ############################
    @staticmethod
    @lru_cache(maxsize=None)
    def __get_metric_size_dict(graph: nx.DiGraph) -> Dict[str, int]:
        ret = {}
        for _, data in graph.nodes(data=True):
            if data['type'] not in ret:
                ret[data['type']] = data['metrics']
            else:
                assert list(map(
                    lambda _: _.split('##')[1],
                    ret[data['type']]
                )) == list(map(
                    lambda _: _.split('##')[1],
                    data['metrics']
                )), \
                    f"The metrics should be same for the nodes of each type: " \
                    f"{data['type']=} {ret[data['type']]=} {data['metrics']=}"
        return dict(map(lambda _: (_[0], len(_[1])), ret.items()))

    def __get_hg(self, graph: nx.DiGraph) -> DGLHeteroGraph:
        hg_data_dict = defaultdict(lambda: ([], []))
        for u, v, data in graph.edges(data=True):
            u_type = graph.nodes[u]['type']
            v_type = graph.nodes[v]['type']
            edge_type = (u_type, f"{u_type}-{data['type']}-{v_type}", v_type)
            hg_data_dict[edge_type][0].append(self._node_to_idx[u_type][u])
            hg_data_dict[edge_type][1].append(self._node_to_idx[v_type][v])
        num_nodes_dict = {}
        for node_type, v in self._node_list.items():
            num_nodes_dict[node_type] = len(v)
        _hg: DGLHeteroGraph = heterograph(
            {
                key: (th.tensor(src_list).long(), th.tensor(dst_list).long())
                for key, (src_list, dst_list) in hg_data_dict.items()
            },
            num_nodes_dict=num_nodes_dict
        )
        del hg_data_dict
        return _hg

    @staticmethod
    def __get_node_metrics_dict(graph: nx.DiGraph):
        ret = {}
        for node, data in graph.nodes(data=True):
            ret[node] = data['metrics']
            components = {_.split('##')[0] for _ in ret[node]}
            assert len(components) == 1, f"There should be only one component for {node=}. {components=}"
        return ret

    @staticmethod
    def __get_node_type_metrics_dict(
            node_metric_dict: Dict[str, List[str]], nodes_list: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        ret = {}
        for node_type, nodes in nodes_list.items():
            for node in nodes:
                metrics = list(map(lambda _: _.split('##')[1], node_metric_dict[node]))
                if node_type not in ret:
                    ret[node_type] = metrics
                assert ret[node_type] == metrics
        return ret


@lru_cache(maxsize=None)
@profile
def _get_global_id_getter(hg: DGLHeteroGraph) -> Callable[[str, int], int]:
    ptr = 0
    ntype_base_ptr = {}
    for ntype in hg.ntypes:
        if ntype in hg.ntypes:
            ntype_base_ptr[ntype] = ptr
        ptr += hg.number_of_nodes(ntype)
    del ptr

    def getter(node_type: str, node_id: int) -> id:
        return ntype_base_ptr[node_type] + node_id

    return getter


@lru_cache(maxsize=None)
@profile
def _get_global_id_resolver(hg: DGLHeteroGraph) -> Callable[[int], Tuple[str, int]]:
    id_to_type = {}
    ntype_base_ptr = {}
    ptr = 0
    for ntype in hg.ntypes:
        if ntype in hg.ntypes:
            ntype_base_ptr[ntype] = ptr
        new_ptr = ptr + hg.number_of_nodes(ntype)
        for _ in range(ptr, ptr + new_ptr):
            id_to_type[_] = ntype
        ptr = new_ptr
    del ptr, new_ptr

    def resolver(global_id: int) -> Tuple[str, int]:
        global_id = int(global_id)
        node_type = id_to_type[global_id]
        node_id = global_id - ntype_base_ptr[node_type]
        return node_type, node_id

    return resolver
