from text2brick.gym.models.BrickPlacementGNN import BrickPlacementGNN
from text2brick.gym.models.MLPFusion import MLP
from text2brick.gym.models.GNNHead import PositionHead2D
from text2brick.gym.models.SNNImg import SNN

from torch import nn

class Text2Brick_v1(nn.Module):
    def __init__(self, image_target, gnn_node_feature_dim=2, gnn_hidden_dim=64, gnn_output_dim=64, gnn_num_heads=4, mlp_hidden_dim=[128, 64, 32, 16], mlp_output_dim=16, grid_size=(10, 10)): 
        """
        Initializes the text2brick v1.

        Args:
            image_target (torch.Tensor): Target image to compare with, of shape [C, H, W].
            gnn_node_feature_dim (int): Dimensionality of node (brick) input features (e.g., x, y, z, size, rotation).
            gnn_hidden_dim (int): Hidden dimensionality of the GAT layers.
            gnn_output_dim (int): Dimensionality of the output embedding for each node.
            gnn_num_heads (int): Number of attention heads in GAT layers.
            mlp_output_dim (int): Dimensionality of the output of the MLP.
            grid_size (tuple): Size of the 2D grid for position prediction (desired size of the lego world).
        """
        super(Text2Brick_v1, self).__init__()

        print("INIT SNN")
        # Define the Siamese Neural Network model for image comparison
        self.snn = SNN(image_target)  # Outputs a 13x13 feature map
        
        print("INIT GNN")
        # Define the GNN model for graph brick embedding
        self.gnn = BrickPlacementGNN(
            node_feature_dim=gnn_node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_heads=gnn_num_heads
        )

        # Define the MLP for feature fusion and dimensionality reduction
        self.mlp = MLP(
            f_fused_size=(1, 13, 13),  # The size of the fused feature set ie. CNN Output 13x13
            h_graph_size=(1, gnn_output_dim),
            hidden_dims=mlp_hidden_dim
        )
        
        # Define the Position Head for 2D coords prediction
        self.position_head = PositionHead2D(
            mlp_output_dim=(1, mlp_output_dim),
            grid_size=grid_size
        ) 
        
    def forward(self, image_environement, node_features, edge_index, batch=None):
        """
        Forward pass to compute the predicted position of the brick.
        
        Args:
            image_environement (torch.Tensor): Environment image to compare with the target image.
            node_features (torch.Tensor): Node feature matrix (brick features).
            edge_index (torch.Tensor): Edge index tensor defining graph connectivity.
            batch (torch.Tensor, optional): Batch vector for graph processing. None if no batching.
        
        Returns:
            torch.Tensor: Predicted position (2D) for the brick.
        """
        # Get image features by comparing the target and environment images
        image_difference = self.snn(image_environement)
        
        # Get the node embeddings from the GNN
        gnn_embeddings = self.gnn(node_features, edge_index, batch)
                
        # Pass gnn embedded graph and img differences through the MLP to be fused and reduced
        mlp_output = self.mlp(image_difference, gnn_embeddings)
        
        # Predict the position of the brick (either 2D or 3D)
        predicted_position = self.position_head(mlp_output)
        
        return predicted_position
    