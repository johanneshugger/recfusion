import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 activation,
                 dropout,
                 batch_norm,
                 residual=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.residual = residual

        if in_channels != out_channels:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(out_channels)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)

        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = x + h

        h = self.dropout(h)

        return h
