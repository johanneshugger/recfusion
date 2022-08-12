import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse.tensor import SparseTensor
from torch_geometric.nn.models import GCN, GAT, GIN, GraphSAGE
from .graph_layers import GCNLayer
from .mlp import MLP



class GCNModelBase(torch.nn.Module):

    def __init__(self, in_channels,
                 hidden_channels,
                 num_layers,
                 out_channels,
                 dropout,
                 batch_norm,
                 residual):
        super().__init__()
        self.batch_norm = batch_norm
        self.residual = residual

        self.linear = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList(
            [GCNLayer(hidden_channels,
                      hidden_channels,
                      F.relu,
                      dropout,
                      self.batch_norm,
                      self.residual) for _ in range(num_layers-1)])
        self.layers.append(
            GCNLayer(hidden_channels,
                     out_channels,
                     F.relu,
                     dropout,
                     self.batch_norm,
                     self.residual)
        )

    def forward(self, x, edge_index):
        # input embedding
        x = self.linear(x)

        # GCN
        for conv in self.layers:
            x = conv(x, edge_index)

        return x

class ValueGCN(GCNModelBase):
    def __init__(self, in_channels,
                 hidden_channels,
                 num_layers,
                 out_channels,
                 dropout,
                 batch_norm,
                 residual,
                 readout,
                 mlp_in_channels,
                 mlp_hidden_channels,
                 mlp_out_channels=1):
        super().__init__(
            in_channels,
            hidden_channels,
            num_layers,
            out_channels,
            dropout,
            batch_norm,
            residual
        )
        self.readout = readout
        self.mlp = MLP(
            input_dim=mlp_in_channels,
            hidden_dims=mlp_hidden_channels,
            output_dim=mlp_out_channels
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = super().forward(x, edge_index)

        # TODO if data has no batch attribute
        batch = data.batch

        if self.readout == "sum":
            x = global_add_pool(x, batch)
        elif self.readout == "max":
            x = global_max_pool(x, batch)
        elif self.readout == "mean":
            x = global_mean_pool(x, batch)
        else:
            raise NotImplementedError

        data.out = self.mlp(x)
        return data


class PolicyGCN(GCNModelBase):
    def __init__(self, in_channels,
                 hidden_channels,
                 num_layers,
                 out_channels,
                 dropout,
                 batch_norm,
                 residual,
                 mlp_in_channels,
                 mlp_hidden_channels,
                 mlp_out_channels=1):
        super().__init__(
            in_channels,
            hidden_channels,
            num_layers,
            out_channels,
            dropout,
            batch_norm,
            residual
        )
        self.mlp = MLP(
            input_dim=mlp_in_channels,
            hidden_dims=mlp_hidden_channels,
            output_dim=mlp_out_channels
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = super().forward(x, edge_index)
        data.out = self.mlp(x)
        return data


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


class TgGCNReg(TgGCN):
    def __init__(
        self,
        in_features,
        out_features,   
        hidden_feat=256,
        num_layers=2,
        dropout=0.,
    ):
        super().__init__(
            in_features, out_features, hidden_feat, num_layers, dropout
        )


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


