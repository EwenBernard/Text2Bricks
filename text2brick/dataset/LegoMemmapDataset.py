import torch
from torch.utils.data import Dataset
import numpy as np
import torch_geometric
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
import networkx as nx


class LegoMemmapDataset(Dataset):
    def __init__(
        self,
        num_graphs: int,
        memmap_file: str = "./lego_dataset/lego_dataset.dat",
        image_shape: int = 150528,
        max_node_count: int = 140,
    ) -> None:
        """
        Args:
            memmap_file (str): Path to the memory-mapped dataset file.
            image_shape (int): Size of the flattened target and current images.
            max_node_count (int): Maximum number of nodes in the graph.
        """
        self.memmap_file = memmap_file
        self.image_shape = image_shape
        self.max_node_count = max_node_count

        # Calculate the total columns for a single sample
        self.total_columns = (
            image_shape +  # Target image
            pow(max_node_count, 2) +  # Initial node tensor
            max_node_count * (  # Iteration data
                image_shape +  # Current image
                2 +  # Brick coordinates (x, y)
                1 +  # Reward
                pow(max_node_count, 2)
            )
        )

        # Load the memmap in read-only mode
        self.data_memmap = np.memmap(
            self.memmap_file, dtype=np.float32, mode="r",
            shape=(num_graphs, self.total_columns)
        )

        self.samples_per_graph = max_node_count  # Each graph has max_node_count iterations

    def __len__(self) -> int:
        """
        Returns:
            int: Total number of iteration steps in the dataset.
        """
        return self.num_graphs * self.samples_per_graph

    
    def _np_graph_to_torch(self, graph_array: np.array) -> torch_geometric.data.Data:

        # Filter the adjacency matrix to remove padding rows and columns
        adj_matrix = graph_array.reshape(self.max_node_count, self.max_node_count)
        non_zero_indices = np.any(adj_matrix != 0, axis=1)
        filtered_adj_matrix = adj_matrix[non_zero_indices][:, non_zero_indices]

        print(filtered_adj_matrix)

        # Create a NetworkX graph from the filtered adjacency matrix
        graph = nx.from_numpy_array(filtered_adj_matrix)

        print(graph.number_of_nodes())

        if graph.number_of_nodes() == 0:
            # Create an empty Data object
            return Data(x=torch.empty((0, 2), dtype=torch.float), edge_index=torch.empty((2, 0), dtype=torch.long))

        graph = from_networkx(graph)
        graph.x = graph.x.float()
        graph.edge_index = graph.edge_index.long()  

        return graph

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): Index of the iteration step to retrieve.

        Returns:
            tuple: A tuple containing:
                - target_image (torch.Tensor): Target image of the final structure.
                - current_image (torch.Tensor): Current image of the structure being built.
                - brick_coordinates (torch.Tensor): Coordinates of the next brick to add (x, y).
                - reward (float): Reward value for the action.
                - current_graph (torch.Tensor): Graph representation of the current state.
        """
        # Determine which graph and iteration step this index corresponds to
        graph_idx = idx // self.samples_per_graph
        step_idx = idx % self.samples_per_graph

        # Read the entire row corresponding to the graph index
        row_data = self.data_memmap[graph_idx, :]

        # Parse the row data
        data_offset = 0

        # Target image
        target_image = torch.tensor(
            row_data[data_offset:data_offset + self.image_shape], dtype=torch.float32
        )
        data_offset += self.image_shape

        # Skip the initial graph data
        data_offset += pow(self.max_node_count, 2)

        # Skip the data for earlier steps
        iteration_data_size = (
            self.image_shape +  # Current image
            2 +  # Brick coordinates (x, y)
            1 +  # Reward
            pow(self.max_node_count, 2)
        )
        data_offset += step_idx * iteration_data_size

        # Parse the data for the current step
        current_image = torch.tensor(
            row_data[data_offset:data_offset + self.image_shape], dtype=torch.float32
        )
        data_offset += self.image_shape

        brick_coordinates = torch.tensor(
            row_data[data_offset:data_offset + 2], dtype=torch.float32
        )
        data_offset += 2

        reward = row_data[data_offset]
        data_offset += 1
        
        current_graph = self._np_graph_to_torch(row_data[data_offset:data_offset + pow(self.max_node_count, 2)])

        return target_image, current_image, brick_coordinates, reward, current_graph