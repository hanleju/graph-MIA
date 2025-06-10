import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout

from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2, aggr='mean'):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.bn1 = BatchNorm1d(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
        self.bn2 = BatchNorm1d(hidden_channels)

        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
        self.bn3 = BatchNorm1d(hidden_channels)

        self.out_proj = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.out_proj(x)
        return x
    

class DeepGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5, dropout=0.3, aggr='mean'):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # 입력 레이어
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(BatchNorm1d(hidden_channels))

        # 은닉 레이어
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(BatchNorm1d(hidden_channels))

        # 출력 레이어
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        self.bns.append(BatchNorm1d(hidden_channels))
        self.out_proj = Linear(hidden_channels, out_channels)

        self.act = ReLU()
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.act(x)
            x = self.dropout(x)
        return self.out_proj(x)