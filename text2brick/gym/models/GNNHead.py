import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionHead3D(nn.Module): 
    def __init__(self, mlp_output_dim, grid_size=(10, 10, 10)): 
        super(PositionHead3D, self).__init__()
        
        # Position Head: Predicts 3D grid indices for x, y, z
        self.position_x = nn.Sequential(
            nn.Linear(mlp_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, grid_size[0])  # Classification logits for x-axis
        )
        self.position_y = nn.Sequential(
            nn.Linear(mlp_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, grid_size[1])  # Classification logits for y-axis
        )
        self.position_z = nn.Sequential(
            nn.Linear(mlp_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, grid_size[2])  # Classification logits for z-axis
        )

        x_logits = self.position_x(mlp_output_dim)
        y_logits = self.position_y(mlp_output_dim)
        z_logits = self.position_z(mlp_output_dim)
        x = torch.argmax(x_logits, dim=-1)  # Predicted x index
        y = torch.argmax(y_logits, dim=-1)  # Predicted y index
        z = torch.argmax(z_logits, dim=-1)  # Predicted z index
        return torch.stack([x, y, z], dim=-1) # Combine into 3D position tensor
    

class PositionHead2D(nn.Module): 
    def __init__(self, mlp_output_dim, grid_size=(10, 10)): 
        super(PositionHead2D, self).__init__()
        
        # Position Head: Predicts 2D grid indices for x, y
        self.position_x = nn.Sequential(
            nn.Linear(mlp_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, grid_size[0])  # Classification logits for x-axis
        )
        self.position_y = nn.Sequential(
            nn.Linear(mlp_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, grid_size[1])  # Classification logits for y-axis
        )

        x_logits = self.position_x(mlp_output_dim)
        y_logits = self.position_y(mlp_output_dim)
        x = torch.argmax(x_logits, dim=-1)  # Predicted x index
        y = torch.argmax(y_logits, dim=-1)  # Predicted y index
        return torch.stack([x, y], dim=-1) # Combine into 2D position tensor