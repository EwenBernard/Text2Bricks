import torch
import torch.nn as nn
import torch.nn.functional as F

# Add prediction to which node the brick will be connected

class PositionHead3D(nn.Module): 
    def __init__(self, mlp_output_dim, grid_size=(10, 10, 10)): 
        super(PositionHead3D, self).__init__()
        
        # Position Head: Predicts 3D grid indices for x, y, z
        self.position_x = nn.Sequential(
            nn.Linear(mlp_output_dim[1], 128),
            nn.ReLU(),
            nn.Linear(128, grid_size[0])  # Classification logits for x-axis
        )
        self.position_y = nn.Sequential(
            nn.Linear(mlp_output_dim[1], 128),
            nn.ReLU(),
            nn.Linear(128, grid_size[1])  # Classification logits for y-axis
        )
        self.position_z = nn.Sequential(
            nn.Linear(mlp_output_dim[1], 128),
            nn.ReLU(),
            nn.Linear(128, grid_size[2])  # Classification logits for z-axis
        )


    def forward(self, mlp_output):
        """
        Forward pass for position prediction.
        
        Args:
            mlp_output (Tensor): Fused embeddings of shape [batch_size, mlp_output_dim].
        
        Returns:
            Tensor: Predicted grid positions of shape [batch_size, 2] (x, y).
        """
        x_logits = self.position_x(mlp_output)
        y_logits = self.position_y(mlp_output)
        z_logits = self.position_z(mlp_output)
        x = torch.argmax(x_logits, dim=-1)  # Predicted x index
        y = torch.argmax(y_logits, dim=-1)  # Predicted y index
        z = torch.argmax(z_logits, dim=-1)  # Predicted z index
        return torch.stack([x, y, z], dim=-1) # Combine into 3D position tensor
    

class PositionHead2D(nn.Module): 
    def __init__(self, mlp_output_dim, grid_size=(10, 10)): 
        super(PositionHead2D, self).__init__()
        
        # Position Head: Predicts 2D grid indices for x, y
        self.position_x = nn.Sequential(
            nn.Linear(mlp_output_dim[1], 128),
            nn.ReLU(),
            nn.Linear(128, grid_size[0])  # Classification logits for x-axis
        )
        self.position_y = nn.Sequential(
            nn.Linear(mlp_output_dim[1], 128),
            nn.ReLU(),
            nn.Linear(128, grid_size[1])  # Classification logits for y-axis
        )

    def forward(self, mlp_output):
        """
        Forward pass for position prediction.
        
        Args:
            mlp_output (Tensor): Fused embeddings of shape [batch_size, mlp_output_dim].
        
        Returns:
            Tensor: Predicted grid positions of shape [batch_size, 2] (x, y).
        """
        # Compute logits for x and y
        x_logits = self.position_x(mlp_output)
        y_logits = self.position_y(mlp_output)

        # Predict indices using argmax
        x = torch.argmax(x_logits, dim=-1)  # Predicted x index
        y = torch.argmax(y_logits, dim=-1)  # Predicted y index

        # Combine x and y into a single tensor
        return torch.stack([x, y], dim=-1)