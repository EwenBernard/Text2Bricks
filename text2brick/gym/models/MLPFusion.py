from torch import nn
import torch

class MLP(nn.Module):

    def __init__(self, f_fused_size, h_graph_size, hidden_dims=[128, 128, 128, 128]):
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
        self.fc1 = nn.Linear(f_fused_size[0] * f_fused_size[1] + h_graph_size + 1, hidden_dims[0])  # Input size is f_fused_size + h_graph_size + 1 (for reward)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])

    def forward(self, f_fused, h_graph, reward):
        """
        Forward pass of the MLP, where the input feature tensor `f_fused` is flattened
        and combined with the graph tensor `h_graph`. The combined tensor is passed
        through several fully connected layers with ReLU activations.

        Args:
            f_fused (torch.Tensor): The feature tensor, size [batch_size, 13, 13]
            h_graph (torch.Tensor): GNN output tensor, size [batch_size, gnn_hidden_dim] 

        Returns:
            torch.Tensor: The output of the final fully connected layer
        """
        # Flatten tensor starting from the second dimension
        f_fused_flat = f_fused.view(f_fused.size(0), -1).float()  # [batch_size, 13, 13]
        h_graph_flat = h_graph.view(h_graph.size(0), -1).float()  # [batch_size, gnn_output_dim]
        reward = reward.view(-1, 1).float() # [batch_size, 1]

        # Concatenate all inputs along the feature dimension
        try:
            combined_tensor = torch.cat([f_fused_flat, h_graph_flat, reward], dim=1)
        except:
            print(f"Shape of f_fused: {f_fused.shape}")
            print(f"Shape of h_graph: {h_graph.shape}")
            print(f"Shape of reward: {reward.shape}")
            
            print(f"Shape of f_fused_flat: {f_fused_flat.shape}")
            print(f"Shape of h_graph_flat: {h_graph_flat.shape}")
            print(f"Shape of reward_flat: {reward.shape}")
            
            print(h_graph)
            return


        # Pass through the fully connected layers with ReLU activations
        x = self.fc1(combined_tensor)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)

        return x
