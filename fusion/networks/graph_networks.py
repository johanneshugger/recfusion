import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse.tensor import SparseTensor
from torch_geometric.nn.models import GCN, GAT, GIN, GraphSAGE


class GenericTgModel(torch.nn.Module):
    def __init__(self, model, in_features, out_features,
                 hidden_feat=256,
                 num_layers=2,
                 dropout=0.,
                 ):

        super(GenericTgModel, self).__init__()
        self.module = model(in_channels=in_features,
                            hidden_channels=hidden_feat,
                            num_layers=num_layers,
                            out_channels=out_features,
                            dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.module(x, edge_index)
        data.out = x
        return data


class TgGCN(GenericTgModel):
    def __init__(self, in_features, out_features,
                 hidden_feat=256,
                 num_layers=2,
                 dropout=0.,
                 ):
        super().__init__(GCN,
                         in_features=in_features,
                         hidden_feat=hidden_feat,
                         num_layers=num_layers,
                         out_features=out_features,
                         dropout=dropout)


class TgGAT(GenericTgModel):
    def __init__(self, in_features, out_features,
                 hidden_feat=256,
                 num_layers=2,
                 dropout=0.,
                 ):
        super().__init__(GAT,
                         in_features=in_features,
                         hidden_feat=hidden_feat,
                         num_layers=num_layers,
                         out_features=out_features,
                         dropout=dropout)


class TgGIN(GenericTgModel):
    def __init__(self, in_features, out_features,
                 hidden_feat=256,
                 num_layers=2,
                 dropout=0.,
                 ):
        super().__init__(GIN,
                         in_features=in_features,
                         hidden_feat=hidden_feat,
                         num_layers=num_layers,
                         out_features=out_features,
                         dropout=dropout)


class TgGraphSAGE(GenericTgModel):
    def __init__(self, in_features, out_features,
                 hidden_feat=256,
                 num_layers=2,
                 dropout=0.,
                 ):
        super().__init__(GraphSAGE,
                         in_features=in_features,
                         hidden_feat=hidden_feat,
                         num_layers=num_layers,
                         out_features=out_features,
                         dropout=dropout)


