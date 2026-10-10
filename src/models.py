import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, SiLU

from .data_processing import ATOM_FEATURES

def get_activation_function(name):
    if name == 'ReLU':
        return nn.ReLU()
    elif name == 'GELU':
        return nn.GELU()
    elif name == 'Swish':
        return nn.SiLU() # SiLU is the same as Swish
    else:
        raise ValueError(f"Unsupported activation function: {name}")

class GCN(nn.Module):
    def __init__(self, dim_h=256, n_layers=4, dropout_rate=0.1, activation_name='ReLU'):
        super().__init__()
        num_atom_features = sum([len(feats) for feats in ATOM_FEATURES.values()])
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.activation = get_activation_function(activation_name)

        self.layers.append(GCNConv(num_atom_features, dim_h))
        self.bns.append(BatchNorm1d(dim_h))

        for _ in range(n_layers - 2):
            self.layers.append(GCNConv(dim_h, dim_h))
            self.bns.append(BatchNorm1d(dim_h))

        self.layers.append(GCNConv(dim_h, dim_h))
        self.bns.append(BatchNorm1d(dim_h))
        
        self.lin1 = Linear(dim_h, dim_h)
        self.lin2 = Linear(dim_h, 2) # Default for classification

    def set_output_dim(self, dim):
        """Sets the output dimension of the final layer for regression tasks."""
        if dim == 1:
            self.lin2 = Linear(self.lin1.out_features, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, (conv, bn) in enumerate(zip(self.layers, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = global_add_pool(x, batch)
        x = self.lin1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

class GIN(nn.Module):
    def __init__(self, dim_h=256, n_layers=4, dropout_rate=0.1, activation_name='ReLU'):
        super().__init__()
        num_atom_features = sum([len(feats) for feats in ATOM_FEATURES.values()])
        self.dropout_rate = dropout_rate
        self.activation = get_activation_function(activation_name)

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            input_dim = num_atom_features if i == 0 else dim_h
            nn_layer = Sequential(
                Linear(input_dim, dim_h), 
                nn.BatchNorm1d(dim_h), 
                self.activation, 
                Linear(dim_h, dim_h)
            )
            self.layers.append(GINConv(nn_layer))
            self.bns.append(BatchNorm1d(dim_h))

        self.lin1 = Linear(dim_h, dim_h)
        self.lin2 = Linear(dim_h, 2) # Default for classification

    def set_output_dim(self, dim):
        if dim == 1:
            self.lin2 = Linear(self.lin1.out_features, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, (conv, bn) in enumerate(zip(self.layers, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = global_add_pool(x, batch)
        x = self.lin1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

class GAT(nn.Module):
    def __init__(self, dim_h=256, n_layers=4, dropout_rate=0.1, heads=4, activation_name='ReLU'):
        super().__init__()
        num_atom_features = sum([len(feats) for feats in ATOM_FEATURES.values()])
        self.dropout_rate = dropout_rate
        self.activation = get_activation_function(activation_name)
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(GATConv(num_atom_features, dim_h, heads=heads, concat=True, dropout=dropout_rate))
        self.bns.append(BatchNorm1d(dim_h * heads))

        for _ in range(n_layers - 2):
            self.layers.append(GATConv(dim_h * heads, dim_h, heads=heads, concat=True, dropout=dropout_rate))
            self.bns.append(BatchNorm1d(dim_h * heads))

        self.layers.append(GATConv(dim_h * heads, dim_h, heads=1, concat=False, dropout=dropout_rate))
        self.bns.append(BatchNorm1d(dim_h))

        self.lin1 = Linear(dim_h, dim_h)
        self.lin2 = Linear(dim_h, 2) # Default for classification

    def set_output_dim(self, dim):
        if dim == 1:
            self.lin2 = Linear(self.lin1.out_features, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, (conv, bn) in enumerate(zip(self.layers, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = global_add_pool(x, batch)
        x = self.lin1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

def get_model(model_name, **kwargs):
    if model_name == 'GCN':
        return GCN(**kwargs)
    elif model_name == 'GIN':
        return GIN(**kwargs)
    elif model_name == 'GAT':
        return GAT(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
