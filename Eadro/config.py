from failure_dependency_graph import FDGBaseConfig
from typing import List


# class EadroConfig(FDGBaseConfig):
#     alpha: float = 0.5
#     debug: bool = False
#     detect_hiddens: List[int] = [64]
#     locate_hiddens: List[int] = [64]
#     fuse_dim: int = 128
#     metric_hiddens: List[int] = [64]
#     metric_kernel_sizes: List[int] = [2]
#     self_attn: bool = True
#     graph_hiddens: List[int] = [64]
#     graph_attn_head: int=4
#     graph_activation: float=0.2
#     attn_drop: float = 0.
#     init_lr: float = 1e-3
#     max_epoch: int = 500
#     early_stopping_epoch_patience: int = 5
#     valid_epoch_freq: int = 10
#     drop_FDG_edges_fraction: float = 0.

class EadroConfig(FDGBaseConfig):
    alpha: float = 0.5
    debug: bool = False
    detect_hiddens: List[int] = [3]
    locate_hiddens: List[int] = [3]
    fuse_dim: int = 3
    metric_hiddens: List[int] = [3]
    metric_kernel_sizes: List[int] = [2]
    self_attn: bool = True
    graph_hiddens: List[int] = [3]
    graph_attn_head: int=4
    graph_activation: float=0.2
    attn_drop: float = 0.
    init_lr: float = 1e-3
    max_epoch: int = 500
    early_stopping_epoch_patience: int = 5
    valid_epoch_freq: int = 10
    drop_FDG_edges_fraction: float = 0.