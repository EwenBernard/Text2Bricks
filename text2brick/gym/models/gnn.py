import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super(GNN, self).__init__()
        

        # Define the layers: GCNConv layers for graph convolution
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = nn.Linear(out_channels, 1)  # For reward prediction (or other task)

    def forward(self, data):
        # data.x: Node features (brick features)
        # data.edge_index: Graph connectivity in COO format (edges between bricks)
        x, edge_index = data.x, data.edge_index

        # First GCN Layer + ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        
        # Second GCN Layer + ReLU activation
        x = F.relu(self.conv2(x, edge_index))

        # Optional: Global pooling layer (mean aggregation of node features)
        x = torch.mean(x, dim=0, keepdim=True)

        # Output layer (can be used for classification or reward prediction)
        x = self.fc(x)

        return x