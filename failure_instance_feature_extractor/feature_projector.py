from typing import List, Union, Type

import torch as th
import torch.nn.functional as F

from failure_instance_feature_extractor.AE import AEFeatureModule
from failure_instance_feature_extractor.CNN import CNNFeatureModule
from failure_instance_feature_extractor.CNN_AE import CNNAEFeatureModule
from failure_instance_feature_extractor.GRU import GRUFeatureModule
from failure_instance_feature_extractor.GRU_AE import GRUAEFeatureModule
from failure_instance_feature_extractor.GRU_VAE import GRUVAEFeatureModule
from failure_instance_feature_extractor.GRU_Transformer import GRUTransformerFeatureModule


class FIFeatureExtractor(th.nn.Module):
    """
    从每个Failure Instance的指标提取统一的特征
    """
    def __init__(
            self,
            failure_classes: List[str],
            input_tensor_size_list: List[th.Size], feature_size: int,
            feature_projector_type: Union[str, Type[th.nn.Module]] = 'CNN'
    ):
        super().__init__()
        self.node_types = failure_classes
        self.features_projection_module_list = th.nn.ModuleList()

        # 初始化专家网络
        for node_type, input_size in zip(failure_classes, input_tensor_size_list):
            if feature_projector_type == 'CNN':
                self.features_projection_module_list.append(CNNFeatureModule(
                    input_size, feature_size
                ))
            elif feature_projector_type == "GRU":
                self.features_projection_module_list.append(GRUFeatureModule(
                    input_size, feature_size, num_layers=1,
                ))
            elif feature_projector_type == 'AE':
                self.features_projection_module_list.append(AEFeatureModule(input_size, feature_size - 1))
            elif feature_projector_type == 'GRU_AE':
                self.features_projection_module_list.append(GRUAEFeatureModule(input_size, feature_size - 1))
            elif feature_projector_type == 'GRU_VAE':
                self.features_projection_module_list.append(GRUVAEFeatureModule(input_size, feature_size - 1))
            elif feature_projector_type == 'CNN_AE':
                self.features_projection_module_list.append(CNNAEFeatureModule(input_size, feature_size - 1))
            elif feature_projector_type == 'GRU_Transformer':
                self.features_projection_module_list.append(GRUTransformerFeatureModule(input_size, feature_size))
            elif issubclass(feature_projector_type, th.nn.Module):
                self.features_projection_module_list.append(feature_projector_type(input_size, feature_size))
            else:
                raise RuntimeError(f"Unknown {feature_projector_type=}")
        
        if isinstance(feature_projector_type, str) and feature_projector_type.endswith("AE"):
            self.rec_loss = th.zeros(1, 1, 1)

        # 初始化一个transformer编码器层，用于不同节点之间的特征相互作用
        nhead = 3
        assert feature_size % nhead == 0, "feature_size must be divisible by nhead"
        transformer_layer = th.nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead)
        self.transformer_encoder = th.nn.TransformerEncoder(transformer_layer, num_layers=2)

    def forward(self, x: List[th.Tensor]) -> th.Tensor:
        """
        :param x: [type_1_node_features in shape (batch_size, n_metrics, n_timestamps), type_2_nodes_features, ...]
        :return: (batch_size, total_n_nodes, feature_size)
        """
        feat_list = []
        for idx, module in enumerate(self.features_projection_module_list):
            input_x = x[idx]
            projected_feat = module(input_x)  # 每个专家输出 (batch_size, n_nodes, feature_size)
            feat_list.append(projected_feat)
        
        # 将所有专家对应的不同node的特征拼接起来
        combined_feat = th.cat(feat_list, dim=-2)  # (batch_size, total_n_nodes, feature_size)
        
        # 使用transformer对不同node的特征进行交互
        combined_feat = combined_feat.permute(1, 0, 2)  # (total_n_nodes, batch_size, feature_size) - 调整维度以适应Transformer输入
        interacted_feat = self.transformer_encoder(combined_feat)  # (total_n_nodes, batch_size, feature_size)
        interacted_feat = interacted_feat.permute(1, 0, 2)  # (batch_size, total_n_nodes, feature_size) - 调整维度回原始顺序
        
        if hasattr(self, 'rec_loss'):
            self.rec_loss = th.cat([_.rec_loss for _ in self.features_projection_module_list], dim=1)
        
        return interacted_feat