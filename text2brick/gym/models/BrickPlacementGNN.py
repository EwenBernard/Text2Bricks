#GCN -> Graph Convolutional Network  
#https://jonathan-hui.medium.com/graph-convolutional-networks-gcn-pooling-839184205692
#https://arxiv.org/pdf/1901.00596
#https://arxiv.org/pdf/1609.02907

#-> need to predict graph new node at the end 
#-> We can predict x,y coords, ok for simple bricks but Not scalable for large lego piece pool.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class BrickPlacementGNN(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, output_dim, num_heads=4):
        """
        Parameters:
        - node_feature_dim (int): Dimensionality of node input features (e.g., x, y, z, size, rotation).
        - edge_feature_dim (int): Dimensionality of edge input features (e.g., connection type, relative position).
        - hidden_dim (int): Hidden dimensionality of the GAT layers.
        - output_dim (int): Dimensionality of the output embedding for each node.
        - num_heads (int): Number of attention heads in GAT layers.
        """
        super(BrickPlacementGNN, self).__init__()

        # GAT Layer 1
        self.gat1 = GATConv(
            in_channels=node_feature_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            edge_dim=edge_feature_dim,
            concat=True
        )

        # GAT Layer 2
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            edge_dim=edge_feature_dim,
            concat=True
        )

        # Output Layer (to produce latent embeddings for nodes)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for the GNN.

        Parameters:
        - x (Tensor): Node feature matrix of shape [num_nodes, node_feature_dim].
        - edge_index (Tensor): Edge index tensor of shape [2, num_edges], defining graph connectivity.
        - edge_attr (Tensor): Edge feature matrix of shape [num_edges, edge_feature_dim].

        Returns:
        - node_embeddings (Tensor): Latent node embeddings of shape [num_nodes, output_dim].
        """
        # GAT Layer 1 with ReLU activation
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)

        # GAT Layer 2 with ReLU activation
        x = self.gat2(x, edge_index, edge_attr)
        x = F.relu(x)

        # Output layer for latent embeddings
        node_embeddings = self.output_layer(x)

        return node_embeddings