from torch import nn
import torch

class MLP(nn.Module):

    def __init__(self, f_fused_size, h_graph_size, hidden_dims=[128, 64, 32, 16]):
        """
        Initialize the MLP with fixed input sizes for f_fused and h_graph, and customizable hidden layer dimensions.

        Args:
            f_fused_size (int): The flattened size of the f_fused tensor.
            h_graph_size (int): The size of the h_graph tensor (default is 3 for 3D coordinates).
            hidden_dims (list): List of integers specifying the sizes of the hidden layers (default is [128, 64, 32, 16]).
        """
        super(MLP, self).__init__()

        # Store hidden dimensions for later use
        self.hidden_dims = hidden_dims
        # Define the fully connected layers with customizable hidden dimensions
        self.fc1 = nn.Linear(f_fused_size[0] * f_fused_size[1] * f_fused_size[2] + h_graph_size[0] * h_graph_size[1] + 1, hidden_dims[0])  # Input size is f_fused_size + h_graph_size + 1 (for reward)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])

    def forward(self, f_fused, h_graph, reward):
        """
        Forward pass of the MLP, where the input feature tensor `f_fused` is flattened
        and combined with the graph tensor `h_graph`. The combined tensor is passed
        through several fully connected layers with ReLU activations.

        Args:
            f_fused (torch.Tensor): The feature tensor, size [batch_size, C, H, W]
            h_graph (torch.Tensor): GNN output tensor, size [batch_size, gnn_hidden_dim] 

        Returns:
            torch.Tensor: The output of the final fully connected layer
        """
        # Flatten the f_fused tensor starting from the second dimension (C, H, W)
        flatten_tensor_features = f_fused.flatten(start_dim=1)  # Flatten to size [batch_size, C*H*W] -> 13x13 = 169
        # Concatenate the flattened features with the h_graph tensor
        combined_tensor = torch.cat((flatten_tensor_features, h_graph, reward), dim=1)

        # Pass through the fully connected layers with ReLU activations
        x = self.fc1(combined_tensor)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)

        return x
