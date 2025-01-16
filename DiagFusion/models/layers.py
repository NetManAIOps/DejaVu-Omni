import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNClassifier(nn.Module):
    """
    两层GCN+最大池化+线性分类器
    """
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        # self.pool = dglnn.AvgPooling()
        self.pool = dglnn.MaxPooling()
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = self.pool(g, h)
        return self.classify(h)
#         with g.local_scope():
#             g.ndata['h'] = h
#             hg = dgl.mean_nodes(g, 'h')
#             return self.classify(hg)

class RGCNMSL(nn.Module):
    """
    多任务学习：微服务组定位及故障分类"联合训练"
    """
    def __init__(self, in_dim, hidden_dim, out_dim1, out_dim2, etype):
        super(RGCNMSL, self).__init__()
        self.etype = etype
        n_rels = len(set([e.item() for e in etype]))
        self.conv1 = dglnn.RelGraphConv(in_dim, hidden_dim, n_rels)
        self.conv2 = dglnn.RelGraphConv(hidden_dim, hidden_dim, n_rels)
#         self.pool = dglnn.AvgPooling()
        self.pool = dglnn.MaxPooling()
#         self.pool = dglnn.SortPooling(k=2)
        self.cls1 = nn.Linear(hidden_dim, out_dim1)
        self.cls2 = nn.Linear(hidden_dim, out_dim2)

    def forward(self, g, h):
        etype = self.etype.repeat((g.num_edges() // len(self.etype)))
        h = F.relu(self.conv1(g, h, etype))
        h = F.relu(self.conv2(g, h, etype))
        h = self.pool(g, h)
        return self.cls1(h), self.cls2(h)

    def get_embeds(self , g, h):
        etype = self.etype.repeat((g.num_edges() // len(self.etype)))
        h = F.relu(self.conv1(g, h, etype))
#         h = F.dropout(h, p=0.5, training=True)
        h = F.relu(self.conv2(g, h, etype))
#         h = F.dropout(h, p=0.5, training=True)
#         h = self.pool(g, h)
        return h

class RGCNClassifier(nn.Module):
    """
    两层RGCN+最大池化+线性分类器
    """
    def __init__(self, in_dim, hidden_dim, n_classes, etype):
        super(RGCNClassifier, self).__init__()
        self.etype = etype
        n_rels = len(set([e.item() for e in etype]))
        self.conv1 = dglnn.RelGraphConv(in_dim, hidden_dim, n_rels)
        self.conv2 = dglnn.RelGraphConv(hidden_dim, hidden_dim, n_rels)
#         self.pool = dglnn.AvgPooling()
        self.pool = dglnn.MaxPooling()
#         self.pool = dglnn.SortPooling(k=2)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        etype = self.etype.repeat((g.num_edges() // len(self.etype)))
        h = F.relu(self.conv1(g, h, etype))
#         h = F.dropout(h, p=0.5, training=True)
        h = F.relu(self.conv2(g, h, etype))
#         h = F.dropout(h, p=0.5, training=True)
        h = self.pool(g, h)
        return self.classify(h)
#         with g.local_scope():
#             g.ndata['h'] = h
#             hg = dgl.mean_nodes(g, 'h')
#             return self.classify(hg)
    def get_embeds(self , g, h, pool=False):
        etype = self.etype.repeat((g.num_edges() // len(self.etype)))
        h = F.relu(self.conv1(g, h, etype))
#         h = F.dropout(h, p=0.5, training=True)
        h = F.relu(self.conv2(g, h, etype))
#         h = F.dropout(h, p=0.5, training=True)
        if pool:
            h = self.pool(g, h)
        return h

class RGCNv2Classifier(nn.Module):
    """
    两层RGCN+最大池化+线性分类器
    """
    def __init__(self, in_dim, hidden_dim, n_classes, etype):
        super(RGCNv2Classifier, self).__init__()
        self.etype = etype
        n_rels = len(set([e.item() for e in etype]))
        self.conv1 = dglnn.RelGraphConv(in_dim, hidden_dim, n_rels)
        self.conv2 = dglnn.RelGraphConv(hidden_dim, hidden_dim, n_rels)
        # self.pool = dglnn.AvgPooling()
        self.pool = dglnn.MaxPooling()
        self.cls1 = nn.Linear(hidden_dim, n_classes)
#         self.cls2 = nn.Linear(hidden_dim//2, n_classes)

    def forward(self, g, h):
        etype = self.etype.repeat((g.num_edges() // len(self.etype)))
        h = F.relu(self.conv1(g, h, etype))
#         h = F.dropout(h, p=0.5, training=True)
        h = F.relu(self.conv2(g, h, etype))
#         h = F.dropout(h, p=0.5, training=True)
        h = self.pool(g, h)
        h = self.cls1(h)
#         h = self.cls2(h)
#         h = self.cls3(h)
        return h


class GATClassifier(nn.Module):
    """
    两层GAT(第一层GAT不同注意力头的输出之间是拼接操作；第二层是求平均操作)+平均池化+线性分类器
    """
    def __init__(self, in_dim, hidden_dim, n_classes, num_heads):
        super(GATClassifier, self).__init__()
        self.conv1 = dglnn.GATConv(in_dim, hidden_dim, num_heads=num_heads, activation=F.relu) # F.relu
        self.conv2 = dglnn.GATConv(hidden_dim*num_heads, hidden_dim, num_heads=num_heads, activation=F.relu)
        # self.pool = dglnn.AvgPooling()
        self.pool = dglnn.MaxPooling()
        self.classify = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, g, h):
        h = self.conv1(g, h)
        hh = torch.zeros(h.shape[0], h.shape[1]*h.shape[2])
        # 中间层多头特征拼接
        for i in range(h.shape[0]):
            hh[i] = torch.cat([t for t in h[i]])
        h = self.conv2(g, hh)
        # 最后一层多头特征取平均值
        h = torch.mean(h, dim=1)
        h = self.pool(g, h)
        return self.classify(h)
    
    def get_embeds(self , g, h, pool=False):
        h = F.relu(self.conv1(g, h))
        # h = F.dropout(h, p=0.5, training=True)
        h = F.relu(self.conv2(g, h))
        # h = F.dropout(h, p=0.5, training=True)
        if pool:
            h = self.pool(g, h)
        return h

class SAGEClassifier(nn.Module):
    """
    两层SAGEConv+最大池化+线性分类器
    """
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(SAGEClassifier, self).__init__()
        self.conv1 = dglnn.SAGEConv(in_dim, hidden_dim, 'lstm', activation=F.relu) # mean, gcn, pool, lstm
        self.conv2 = dglnn.SAGEConv(hidden_dim, hidden_dim, 'lstm', activation=F.relu)
        # self.pool = dglnn.AvgPooling()
        self.pool = dglnn.MaxPooling()
        self.classify = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, g, h):
        h = self.conv1(g, h)
        h = self.conv2(g, h)
        h = self.pool(g, h)
        return self.classify(h)
    
class TAGClassifier(nn.Module):
    """
    两层TAGConv+最大池化+线性分类器
    """
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(TAGClassifier, self).__init__()
        self.conv1 = dglnn.TAGConv(in_dim, hidden_dim, activation=F.relu)
        self.conv2 = dglnn.TAGConv(hidden_dim, hidden_dim, activation=F.relu)
        # self.pool = dglnn.AvgPooling()
        self.pool = dglnn.MaxPooling()
        self.classify = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, g, h):
        h = self.conv1(g, h)
        h = self.conv2(g, h)
        h = self.pool(g, h)
        return self.classify(h)
    
    def get_embeds(self , g, h, pool=False):
        h = self.conv1(g, h)
        # h = F.dropout(h, p=0.5, training=True)
        h = self.conv2(g, h)
        # h = F.dropout(h, p=0.5, training=True)
        if pool:
            h = self.pool(g, h)
        return h

class GATv2Classifier(nn.Module):
    """
    两层GATv2+平均池化+线性分类器
    """
    def __init__(self, in_dim, hidden_dim, n_classes, num_heads):
        super(GATv2Classifier, self).__init__()
        self.conv1 = dglnn.GATv2Conv(in_dim, hidden_dim, num_heads=num_heads, activation=F.relu) # F.relu
        self.conv2 = dglnn.GATv2Conv(hidden_dim*num_heads, hidden_dim, num_heads=num_heads, activation=F.relu)
        # self.pool = dglnn.AvgPooling()
        self.pool = dglnn.MaxPooling()
        self.classify = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, g, h):
        h = self.conv1(g, h)
        hh = torch.zeros(h.shape[0], h.shape[1]*h.shape[2])
        # 中间层多头特征拼接
        for i in range(h.shape[0]):
            hh[i] = torch.cat([t for t in h[i]])
        h = self.conv2(g, hh)
        # 最后一层多头特征取平均值
        h = torch.mean(h, dim=1)
        h = self.pool(g, h)
        return self.classify(h)

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(LinearClassifier, self).__init__()
        # self.pool = dglnn.AvgPooling()
        self.pool = dglnn.MaxPooling()
        # self.linear1 = nn.Linear(in_dim, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.classify = nn.Linear(hidden_dim, n_classes)
        self.classify = nn.Linear(in_dim, n_classes)

    def forward(self, g, h):
        h = self.pool(g, h)
        # h = F.relu(self.linear1(h))
        # h = F.relu(self.linear2(h))
        return self.classify(h)
    
class SGCCClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(SGCCClassifier, self).__init__()
        self.conv1 = dglnn.SGConv(in_dim, hidden_dim)
        self.conv2 = dglnn.SGConv(hidden_dim, hidden_dim)
        # self.pool = dglnn.AvgPooling()
        self.pool = dglnn.MaxPooling()
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = self.pool(g, h)
        return self.classify(h)

class ChebClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, k=2, compute_lambda_max=False):
        super(ChebClassifier, self).__init__()
        self.conv1 = dglnn.ChebConv(in_dim, hidden_dim, k, activation=F.relu)
        self.conv2 = dglnn.ChebConv(hidden_dim, hidden_dim, k, activation=F.relu)
        # self.pool = dglnn.AvgPooling()
        self.pool = dglnn.MaxPooling()
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.lambda_max = None
        self.compute_lambda_max = compute_lambda_max
    
    def forward(self, g, h):
        if self.compute_lambda_max:
#             self.lambda_max = dgl.laplacian_lambda_max(g)
            if self.lambda_max is None:
                self.lambda_max = dgl.laplacian_lambda_max(g)
        else:
            self.labda_max = 2
        try:
            h = self.conv1(g, h, self.lambda_max)
        except:
            self.lambda_max = dgl.laplacian_lambda_max(g)
            h = self.conv1(g, h, self.lambda_max)
        h = self.conv2(g, h, self.lambda_max)
        h = self.pool(g, h)
        return self.classify(h)