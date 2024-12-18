from torch.utils.data import Dataset
import numpy as np
import random
import torch
from torch_geometric.utils.convert import from_networkx

from text2brick.models import GraphLegoWorldData
from text2brick.dataset.dataset import MNISTDataset


class CustomDatasetGraph(Dataset):
    
    def __init__(self):
        self.mnist = MNISTDataset()
        self.length = 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Fetch an item from the dataset given an index. This includes sampling data from MNIST, 
        generating a subgraph, processing the graph data, and returning it as PyTorch tensors.

        Args:
            index (int): The index of the sample to retrieve. In this case, index is unused.

        Returns:
            tuple: A tuple containing:
                - node_tensor (Tensor): Tensor containing node features (x, y coordinates).
                - edge_index (Tensor): Tensor containing the indices of the graph's edges.
                - next_node (Tensor): Tensor containing the coordinates of the next node (brick) to place.
        """
        array, _, _, _ = self.mnist.sample()
        lego_world_ref = GraphLegoWorldData(array)
        random_index = random.randint(0, lego_world_ref.nodes_num())

        # Generate subgraph and next node based on random index
        if random_index == lego_world_ref.nodes_num():
            new_graph = lego_world_ref.graph
            next_node = np.array([-1, -1])  # No next brick to place
        else:
            new_graph = lego_world_ref.subgraph(random_index + 1)
            last_node = max(new_graph.nodes)
            last_node_data = new_graph.nodes[last_node]
            new_graph.remove_node(last_node)  # Remove the last brick
            next_node = np.array([last_node_data['x'], last_node_data['y']])

        # Convert to graph data and process
        new_graph_data = from_networkx(new_graph)

        # Handle empty graph case directly
        node_tensor = torch.stack([new_graph_data.x, new_graph_data.y]) if new_graph_data.x is not None and new_graph_data.y is not None else torch.empty(2, 0, dtype=torch.int)

        edge_index = torch.unique(torch.sort(new_graph_data.edge_index, dim=0)[0], dim=1)  # Sort and remove duplicate edges
        next_node = torch.tensor(next_node, dtype=torch.int)

        return node_tensor, edge_index, next_node