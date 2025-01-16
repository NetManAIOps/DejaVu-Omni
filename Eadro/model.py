import torch
from torch import nn
from dgl.nn.pytorch import GATv2Conv
from dgl.nn import GlobalAttentionPooling
import numpy as np
import math
import dgl

from typing import List
from pyprof import profile
from einops import rearrange
from torch.utils.data import DataLoader

from failure_dependency_graph import FDG
from metric_preprocess import MetricPreprocessor
from DejaVu.dataset import DejaVuDataset
from Eadro.config import EadroConfig
from failure_dependency_graph import FDG, split_failures_by_type, split_failures_by_drift


class GraphModel(nn.Module):
    def __init__(self, in_dim, config:EadroConfig, device='cpu'):
        super(GraphModel, self).__init__()
        '''
        Params:
            in_dim: the feature dim of each node
        '''
        layers = []

        for i, hidden in enumerate(config.graph_hiddens):
            in_feats = config.graph_hiddens[i-1] if i > 0 else in_dim 
            layers.append(GATv2Conv(in_feats, out_feats=hidden, num_heads=config.graph_attn_head, 
                                        attn_drop=config.attn_drop, negative_slope=config.graph_activation, allow_zero_in_degree=True)) 
            self.maxpool = nn.MaxPool1d(config.graph_attn_head)

        self.net = nn.Sequential(*layers).to(device)
        self.out_dim = config.graph_hiddens[-1]
        self.pooling = GlobalAttentionPooling(nn.Linear(self.out_dim, 1)) 

    
    def forward(self, graph, x):
        '''
        Input:
            x -- tensor float [batch_size*node_num, feature_in_dim] N = {s1, s2, s3, e1, e2, e3}
        '''
        out = None
        for layer in self.net:
            if out is None: out = x
            out = layer(graph, out)
            out = self.maxpool(out.permute(0, 2, 1)).permute(0, 2, 1).squeeze()
        return self.pooling(graph, out) #[bz*node, out_dim] --> [bz, out_dim]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes, dilation=2, dev="cpu"):
        super(ConvNet, self).__init__()
        layers = []
        for i in range(len(kernel_sizes)):
            dilation_size = dilation ** i
            kernel_size = kernel_sizes[i]
            padding = (kernel_size-1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding), 
                       nn.BatchNorm1d(out_channels), nn.ReLU(), Chomp1d(padding)]
            
        self.network = nn.Sequential(*layers)
        
        self.out_dim = num_channels[-1]
        self.network.to(dev)
        
    
    def forward(self, x): #[batch_size, n_node, T, in_dim]
        batch_size, n_node, T, in_dim = x.shape
        x = x.permute(0, 1, 3, 2).float() #[batch_size, n_node, in_dim, T]
        out = self.network(x.view((-1, x.shape[-2], x.shape[-1]))) #[batch_size * n_node, out_dim, T]
        out = out.permute(0, 2, 1) #[batch_size * n_node, T, out_dim]
        return out.view((batch_size, n_node, out.shape[-2], out.shape[-1]))


class SelfAttention(nn.Module):
    def __init__(self, input_size, seq_len):
        """
        Args:
            input_size: int, hidden_size * num_directions
            seq_len: window_size
        """
        super(SelfAttention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.atten_bias.data.fill_(0)

    def forward(self, x):
        # x: [batch_size, window_size, input_size]
        input_tensor = x.transpose(1, 0)  # w x b x h
        input_tensor = (torch.bmm(input_tensor, self.atten_w) + self.atten_bias)  # w x b x out
        input_tensor = input_tensor.transpose(1, 0)
        atten_weight = input_tensor.tanh()
        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), x).squeeze()
        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)


class MetricModel(nn.Module):
    def __init__(self, metric_num_list, chunk_length, config:EadroConfig, device='cpu'):
        super(MetricModel, self).__init__()
        metric_hiddens = list(config.metric_hiddens)
        metric_kernel_sizes = list(config.metric_kernel_sizes)
        assert len(metric_hiddens) == len(metric_kernel_sizes)

        self.metric_num_list = metric_num_list
        self.out_dim = metric_hiddens[-1]

        self.net_list = []
        for metric_num in metric_num_list:
            self.net_list.append(ConvNet(num_inputs=metric_num, num_channels=metric_hiddens,
                kernel_sizes=metric_kernel_sizes, dev=device))

        self.self_attn = config.self_attn
        if self.self_attn:
            assert (chunk_length is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_length)

    def forward(self, graphs: List[dgl.DGLGraph], x: List[torch.Tensor]):
        """
        :param x: [type_1_node_features in shape (n_nodes, n_timestamps, n_metrics), type_2_nodes_features, ]
        :return: (n_nodes, feature_size)
        """
        feat_list = []
        for idx, net in enumerate(self.net_list):
            input_x = x[idx]
            feat_list.append(net(input_x))
        feat = torch.cat(feat_list, dim=-3)  # (batch_size, N, T, out_dim)
        # select the features of the instances that are retained in the graphs
        feat = torch.cat([feat[i, g.ndata[dgl.NID]] for i, g in enumerate(graphs)])
        if self.self_attn: 
            return self.attn_layer(feat)
        return feat[:,-1,:] #[bz, out_dim]


class MultiSourceEncoder(nn.Module):
    def __init__(self, metric_num, node_num, chunk_length, device, config:EadroConfig):
        super(MultiSourceEncoder, self).__init__()
        self.node_num = node_num
        self.alpha = config.alpha

        trace_dim,log_dim = 0, 0
        self.metric_model = MetricModel(metric_num, chunk_length=chunk_length, config=config, device=device)
        metric_dim = self.metric_model.out_dim
        fuse_in = trace_dim+log_dim+metric_dim

        fuse_dim = config.fuse_dim
        if not fuse_dim % 2 == 0: fuse_dim += 1
        self.fuse = nn.Linear(fuse_in, fuse_dim)

        self.activate = nn.GLU()
        self.feat_in_dim = int(fuse_dim // 2)

        
        self.status_model = GraphModel(in_dim=self.feat_in_dim, config=config, device=device)
        self.feat_out_dim = self.status_model.out_dim
    
    def forward(self, graphs, feats):
        metric_embedding = self.metric_model(graphs, feats) #[bz*node_num, metric_dim]
        # [bz*node_num, fuse_in] --> [bz, fuse_out], fuse_in: sum of dims from multi sources
        feature = self.activate(self.fuse(metric_embedding)) #[bz*node_num, node_dim]

        batch_graph = dgl.batch(graphs)
        batch_graph.ndata['graph_id'] = dgl.broadcast_nodes(
            batch_graph, torch.arange(len(graphs), device=batch_graph.device)[:, None]
        )[:, 0]
        embeddings = self.status_model(batch_graph, feature) #[bz, graph_dim]
        return embeddings


class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, linear_sizes):
        super(FullyConnected, self).__init__()
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i-1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor): #[batch_size, in_dim]
        return self.net(x)


class MainModel(nn.Module):
    def __init__(self, metric_num_list, node_num, chunk_length, device, config:EadroConfig):
        super(MainModel, self).__init__()

        self.device = device
        self.node_num = node_num
        self.alpha = config.alpha

        self.encoder = MultiSourceEncoder(metric_num_list, node_num, chunk_length, device, config)
        self.detector = FullyConnected(self.encoder.feat_out_dim, 2, config.detect_hiddens).to(device)
        self.detector_criterion = nn.CrossEntropyLoss()
        self.localizer = FullyConnected(self.encoder.feat_out_dim, node_num, config.locate_hiddens).to(device)
        self.localizer_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.get_prob = nn.Softmax(dim=-1)

    def forward(self, graphs, feats, fault_indexs):
        batch_size = len(graphs)
        embeddings = self.encoder(graphs, feats) #[bz, feat_out_dim]
        
        y_prob = torch.zeros((batch_size, self.node_num)).to(self.device) 
        for i in range(batch_size):
            if fault_indexs[i] > -1: 
                y_prob[i, fault_indexs[i]] = 1
        y_anomaly = torch.zeros(batch_size).long().to(self.device)
        for i in range(batch_size):
            y_anomaly[i] = int(fault_indexs[i] > -1)


        locate_logits = self.localizer(embeddings)
        locate_loss = self.localizer_criterion(locate_logits, fault_indexs.to(self.device))
        detect_logits = self.detector(embeddings)
        detect_loss = self.detector_criterion(detect_logits, y_anomaly) 
        loss = self.alpha * detect_loss + (1-self.alpha) * locate_loss

        node_probs = self.get_prob(locate_logits.detach()).cpu().numpy()
        y_pred = self.inference(batch_size, node_probs, detect_logits)
        
        return {'loss': loss, 'y_pred': y_pred, 'y_prob': y_prob.detach().cpu().numpy(), 'pred_prob': node_probs}
        
    def inference(self, batch_size, node_probs, detect_logits=None):
        node_list = np.flip(node_probs.argsort(axis=1), axis=1)
        
        y_pred = []
        for i in range(batch_size):
            detect_pred = detect_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
            if detect_pred[i] < 1: y_pred.append([-1])
            else: y_pred.append(node_list[i])
        
        return y_pred


class EadroModel:
    def __init__(self, cdp: FDG, mp: MetricPreprocessor, config: EadroConfig, device='cuda'):
        self.config = config
        self.fdg = cdp
        self.device = device
        self.drift_list = []
        if config.dataset_split_method == 'type':
            self.train_fault_ids, self.validation_fault_ids, self.test_fault_ids = split_failures_by_type(
                self.fdg.failures_df, split=config.dataset_split_ratio,
                train_set_sampling_ratio=config.train_set_sampling,
                balance_train_set=config.balance_train_set,
                fdg=self.fdg,
            )
        elif config.dataset_split_method == 'drift':
            self.train_fault_ids, self.validation_fault_ids, self.test_fault_ids, self.drift_list = split_failures_by_drift(
                self.fdg.failures_df, split=config.dataset_split_ratio,
                train_set_sampling_ratio=config.train_set_sampling,
                balance_train_set=config.balance_train_set,
                fdg=self.fdg, drift_time=config.drift_time,
            )
        self.train_dataset = DejaVuDataset(
            cdp=self.fdg,
            feature_extractor=mp,
            fault_ids=self.train_fault_ids,
            window_size=self.config.window_size,
            augmentation=False,
            drop_edges_fraction=self.config.drop_FDG_edges_fraction,
            device=self.device,
        )
        self.valid_dataset = DejaVuDataset(
            cdp=self.fdg,
            feature_extractor=mp,
            fault_ids=self.validation_fault_ids,
            window_size=self.config.window_size,
            augmentation=False,
            drop_edges_fraction=self.config.drop_FDG_edges_fraction,
            device=self.device,
        )
        self.test_dataset = DejaVuDataset(
            cdp=self.fdg,
            feature_extractor=mp,
            fault_ids=self.test_fault_ids,
            window_size=self.config.window_size,
            augmentation=False,
            drop_edges_fraction=self.config.drop_FDG_edges_fraction,
            device=self.device,
        )
        metric_num_list = [self.fdg.metric_number_dict[_] for _ in self.fdg.failure_classes]
        print(f"{metric_num_list=}")
        node_num = self.fdg.n_failure_instances
        chunk_length = config.window_size[0] + config.window_size[1]
        self.module = MainModel(metric_num_list, node_num, chunk_length, device, config)
        self.setup_dataloader()
    
    def get_collate_fn(self, batch_size: int):
        if batch_size is None:
            @profile
            def collate_fn(batch_data):
                features_list, label, failure_id, graph = batch_data
                features_list = [rearrange(item, "N F T -> N T F") for item in features_list]  # n_nodes, n_timestamps, n_metrics
                faultidx = torch.argmax(label, dim=-1)
                return [v.type(torch.float32) for v in features_list], faultidx, failure_id, graph
        else:
            @profile
            def collate_fn(batch_data):
                feature_list_list, labels_list, failure_id_list, graph_list = tuple(map(list, zip(*batch_data)))
                features_list = list(map(lambda _: torch.stack(_).float(), zip(*feature_list_list)))
                features_list = [rearrange(item, "B N F T -> B N T F") for item in features_list]  # n_nodes, n_timestamps, n_metrics
                labels = torch.stack(labels_list, dim=0)
                faultidxs = torch.argmax(labels, dim=-1)
                return features_list, faultidxs, torch.tensor(failure_id_list), graph_list
        return collate_fn
    
    def setup_dataloader(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size,
            collate_fn=self.get_collate_fn(self.config.batch_size),
            shuffle=True,
            pin_memory=False,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.config.test_batch_size,
            collate_fn=self.get_collate_fn(self.config.batch_size),
            shuffle=False,
            pin_memory=False,
        )
        self.val_dataloader = DataLoader(
            self.valid_dataset, batch_size=self.config.test_batch_size,
            collate_fn=self.get_collate_fn(self.config.batch_size),
            shuffle=False,
            pin_memory=False,
        )

    def set_device(self, device):
        self.module.to(device)
        self.train_dataset.to(device)
        self.valid_dataset.to(device)
        self.test_dataset.to(device)
    