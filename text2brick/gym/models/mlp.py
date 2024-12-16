from torch import nn
import torch

from text2brick.models import Brick


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
    

    def forward(self, f_fused, h_graph):
        """
        Forward pass of the MLP, where the input feature tensor `f_fused` is flattened
        and combined with the graph tensor `h_graph`. The combined tensor is passed
        through several fully connected layers with ReLU activations.

        Args:
            f_fused (torch.Tensor): The feature tensor, typically of size [batch_size, C, H, W]
            h_graph (torch.Tensor): The graph coordinates tensor, typically of size [batch_size, 3]

        Returns:
            torch.Tensor: The output of the final fully connected layer
        """
        flatten_tensor_features = f_fused.flatten(start_dim=0) # 13x13 = 169
        combined_tensor = torch.cat((flatten_tensor_features, h_graph))

        data_size = combined_tensor.size()[0]

        fc1 = nn.Linear(data_size, 128)
        fc2 = nn.Linear(128, 64)
        fc3 = nn.Linear(64, 32)
        fc4 = nn.Linear(32, 16)

        x = fc1(combined_tensor)
        x = torch.relu(x)
        x = fc2(x)
        x = torch.relu(x)
        x = fc3(x)
        x = torch.relu(x)
        x = fc4(x)

        return x

