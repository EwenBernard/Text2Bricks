#GCN -> Graph Convolutional Network  
#https://jonathan-hui.medium.com/graph-convolutional-networks-gcn-pooling-839184205692
#https://arxiv.org/pdf/1901.00596
#https://arxiv.org/pdf/1609.02907

#-> need to predict graph new node at the end 
#-> We can predict x,y coords, ok for simple bricks but Not scalable for large lego piece pool.

# add edge embedding processing in the V2 version

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class BrickPlacementGNN(nn.Module):
    def __init__(self, node_feature_dim=2, hidden_dim=64, output_dim=64, num_heads=4):
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
            concat=True
        )

        # GAT Layer 2
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True
        )

        # Output Layer (to produce latent embeddings for nodes)
        self.output_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, edge_index, batch=None):
        """
        Forward pass for the GNN.

        Parameters:
        - x (Tensor): Node feature matrix of shape [num_nodes, node_feature_dim].
        - edge_index (Tensor): Edge index tensor of shape [2, num_edges], defining graph connectivity.
        - edge_attr (Tensor): Edge feature matrix of shape [num_edges, edge_feature_dim].
        - batch (Tensor): Batch vector, which assigns each node in the graph to a specific graph in the batch. None if no batch

        Returns:
        - graph_embedding (Tensor): A fixed-size graph-level embedding size [batch_size, output_dim].
        """
        # GAT Layer 1 with ReLU activation
        x = self.gat1(x, edge_index)
        x = F.relu(x)

        # GAT Layer 2 with ReLU activation
        x = self.gat2(x, edge_index)
        x = F.relu(x)

        # Output layer for latent node embeddings
        node_embeddings = self.output_layer(x)

        # Global mean pooling to aggregate node embeddings to a graph-level embedding
        graph_embedding = global_mean_pool(node_embeddings, batch) # shape: [batch_size, output_dim]

        return graph_embedding
