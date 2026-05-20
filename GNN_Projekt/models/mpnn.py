import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool

class EdgeMPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add')
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out):
        return self.node_mlp(aggr_out)

class MPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, hidden_layer_dim, dropout=0.3):
        super().__init__()
        self.conv1 = EdgeMPNNLayer(node_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeMPNNLayer(hidden_dim, edge_dim, hidden_dim)
        self.conv3 = EdgeMPNNLayer(hidden_dim, edge_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

        self.hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_layer_dim),
            nn.ReLU()
        )
        self.output = nn.Linear(hidden_layer_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x.float()
        edge_attr = edge_attr.float()

        x = self.bn1(self.conv1(x, edge_index, edge_attr))
        x = self.bn2(self.conv2(x, edge_index, edge_attr))
        x = self.bn3(self.conv3(x, edge_index, edge_attr))

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.output(x)
        return x.squeeze(-1)

class MPNNAdd(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, hidden_layer_dim, dropout=0.3):
        super().__init__()
        self.conv1 = EdgeMPNNLayer(node_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeMPNNLayer(hidden_dim, edge_dim, hidden_dim)
        self.conv3 = EdgeMPNNLayer(hidden_dim, edge_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

        self.hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_layer_dim),
            nn.ReLU()
        )
        self.output = nn.Linear(hidden_layer_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x.float()
        edge_attr = edge_attr.float()

        x = self.bn1(self.conv1(x, edge_index, edge_attr))
        x = self.bn2(self.conv2(x, edge_index, edge_attr))
        x = self.bn3(self.conv3(x, edge_index, edge_attr))

        x = global_add_pool(x, batch)
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.output(x)
        return x.squeeze(-1)


class MPNNAdd5Layers(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, hidden_layer_dim, dropout=0.3):
        super().__init__()
        self.conv1 = EdgeMPNNLayer(node_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeMPNNLayer(hidden_dim, edge_dim, hidden_dim)
        self.conv3 = EdgeMPNNLayer(hidden_dim, edge_dim, hidden_dim)
        self.conv4 = EdgeMPNNLayer(hidden_dim, edge_dim, hidden_dim)
        self.conv5 = EdgeMPNNLayer(hidden_dim, edge_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

        self.hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_layer_dim),
            nn.ReLU()
        )
        self.output = nn.Linear(hidden_layer_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x.float()
        edge_attr = edge_attr.float()

        x = self.bn1(self.conv1(x, edge_index, edge_attr))
        x = self.bn2(self.conv2(x, edge_index, edge_attr))
        x = self.bn3(self.conv3(x, edge_index, edge_attr))
        x = self.bn4(self.conv4(x, edge_index, edge_attr))
        x = self.bn5(self.conv5(x, edge_index, edge_attr))

        x = global_add_pool(x, batch)
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.output(x)
        return x.squeeze(-1)

class MPNNMultiPool(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, hidden_layer_dim, dropout=0.3):
        super().__init__()
        self.conv1 = EdgeMPNNLayer(node_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeMPNNLayer(hidden_dim, edge_dim, hidden_dim)
        self.conv3 = EdgeMPNNLayer(hidden_dim, edge_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

        # We concatenate 3 poolings (mean, max, add), so the input is 3 * hidden_dim
        self.hidden = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_layer_dim),
            nn.ReLU()
        )
        self.output = nn.Linear(hidden_layer_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x.float()
        edge_attr = edge_attr.float()

        x = self.bn1(self.conv1(x, edge_index, edge_attr))
        x = self.bn2(self.conv2(x, edge_index, edge_attr))
        x = self.bn3(self.conv3(x, edge_index, edge_attr))

        # Multi-Pooling Readout
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x_add  = global_add_pool(x, batch)
        
        # Concatenate all three poolings into one long vector
        x = torch.cat([x_mean, x_max, x_add], dim=1)

        x = self.dropout(x)
        x = self.hidden(x)
        x = self.output(x)
        return x.squeeze(-1)
