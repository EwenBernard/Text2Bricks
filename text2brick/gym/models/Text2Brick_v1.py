from text2brick.gym.models.BrickPlacementGNN import BrickPlacementGNN
from text2brick.gym.models.MLPFusion import MLP
from text2brick.gym.models.GNNHead import PositionHead2D
from text2brick.gym.models.SNNImg import SNN

from torch import nn

class Text2Brick_v1(nn.Module):
    def __init__(
            self,
            image_target=None,
            grid_size=(28, 28),
            gnn_node_feature_dim=2,
            gnn_hidden_dim=64,
            gnn_output_dim=64,
            mlp_hidden_dim=[128, 128, 128, 128],
            mlp_output_dim=128,
            *args,
            **kwargs
            ) -> None: 
        """
        Initializes the text2brick v1.

        Args:
            image_target (torch.Tensor): Target image to compare with, of shape [C, H, W].
            gnn_node_feature_dim (int): Dimensionality of node (brick) input features (e.g., x, y, z, size, rotation).
            gnn_hidden_dim (int): Hidden dimensionality of the GAT layers.
            gnn_output_dim (int): Dimensionality of the output embedding for each node.
            mlp_output_dim (int): Dimensionality of the output of the MLP.
            grid_size (tuple): Size of the 2D grid for position prediction (desired size of the lego world).
        """
        super(Text2Brick_v1, self).__init__()

        # Define the Siamese Neural Network model for image comparison
        self.snn = SNN(
            image_target=image_target,
            normalize=kwargs.get("normalize", False),               # Post process option
            amplification=kwargs.get("amplification", False),       # Post process option
            threshold_factor=kwargs.get("threshold_factor", 0.0),   # Post process option
            squeeze_output=kwargs.get("squeeze_output", False)      # Post process option
        )
        
        # Define the GNN model for graph brick embedding
        self.gnn = BrickPlacementGNN(
            node_feature_dim=gnn_node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim
        )

        # Define the MLP for feature fusion and dimensionality reduction
        self.mlp = MLP(
            f_fused_size=self.snn.ouput_size,  # The size of the fused feature set ie. CNN Output 13x13
            h_graph_size=gnn_output_dim,
            hidden_dims=mlp_hidden_dim,
            dropout_prob=kwargs.get("dropout_prob", 0.0)
        )
        
        # Define the Position Head for 2D coords prediction
        self.position_head = PositionHead2D(
            mlp_output_dim=mlp_output_dim,
            grid_size=grid_size
        ) 

    def forward(self, image_environement, gnn_input, reward, image_target=None, return_logits=False):
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
        image_difference = self.snn(image_environement, image_target=image_target)
        
        # Get the node embeddings from the GNN
        gnn_embeddings = self.gnn(gnn_input.x, gnn_input.edge_index, gnn_input.batch)

        # Pass gnn embedded graph and img differences through the MLP to be fused and reduced
        mlp_output = self.mlp(image_difference, gnn_embeddings, reward)
        
        # Predict the position of the brick (either 2D or 3D)
        predicted_position = self.position_head(mlp_output, return_logits=return_logits)

        return predicted_position