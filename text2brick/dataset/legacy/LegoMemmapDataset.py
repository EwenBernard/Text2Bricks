import torch
from torch.utils.data import Dataset
import numpy as np
import torch_geometric
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
import networkx as nx
from text2brick.dataset.Preprocessing import PreprocessImage

PADDING_VALUE = -1


class LegoMemmapDataset(Dataset):
    def __init__(
        self,
        num_graphs: int,
        memmap_file: str = "./lego_dataset/lego_dataset.dat",
        image_shape: int = 784,
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
        self.max_possible_edges = pow(max_node_count, 2)
        self.preprocess_image = PreprocessImage()

        # Calculate the total columns for a single sample with new format
        self.total_columns = (
            image_shape +  # Target image
            (2 * self.max_possible_edges) +  # Edge indices (flattened [num_edges, 2] array)
            max_node_count +  # Node values
            max_node_count * (  # Iteration data
                image_shape +  # Current image
                2 +  # Brick coordinates (x, y)
                1 +  # Reward
                (2 * self.max_possible_edges) +  # Edge indices for current state
                (2 * max_node_count)  # Node values for current state
            )
        )

        # Load the memmap in read-only mode
        self.data_memmap = np.memmap(
            self.memmap_file, dtype=np.float32, mode="r",
            shape=(num_graphs, self.total_columns)
        )

        self.samples_per_graph = max_node_count
        self.num_graphs = num_graphs

    def __len__(self) -> int:
        """
        Returns:
            int: Total number of iteration steps in the dataset.
        """
        return self.num_graphs * self.samples_per_graph

    def _process_graph_data(self, edge_index_flat, node_values) -> Data:
        """
        Convert flattened edge indices and node values to PyG Data object.
        """
        # Reshape edge indices to [num_edges, 2]
        edge_indices = edge_index_flat.reshape(-1, 2)
        
        # Filter out padding (-1 values)
        valid_edges_mask = ~(edge_indices == PADDING_VALUE).any(axis=1)
        valid_edges = edge_indices[valid_edges_mask]

        # print(valid_edges)
        # print(valid_edges.shape)
        
        # Filter out padding in node values
        valid_nodes_mask = node_values != PADDING_VALUE
        valid_nodes = node_values[valid_nodes_mask].reshape(-1, 2)

        # print(valid_nodes)
        # print(valid_nodes.shape)

        # If no valid nodes/edges, return empty graph
        if len(valid_nodes) == 0:
            return Data(
                x=torch.empty((0, 2), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long)
            )

        # Convert to PyG format
        edge_index = torch.tensor(valid_edges.T, dtype=torch.long)  # Convert to [2, num_edges]
        node_features = torch.tensor(valid_nodes, dtype=torch.float32)

        return Data(x=node_features, edge_index=edge_index)

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): Index of the iteration step to retrieve.

        Returns:
            tuple: (target_image, current_image, brick_coordinates, reward, current_graph)
        """
        graph_idx = idx // self.samples_per_graph
        step_idx = idx % self.samples_per_graph

        # Read the entire row
        row_data = self.data_memmap[graph_idx, :]
        data_offset = 0
        
        target_image = np.array(np.reshape(row_data[data_offset:data_offset + self.image_shape], (28, 28)))
        processed_target_image = self.preprocess_image(target_image)

        data_offset += self.image_shape

        # Skip initial graph data
        data_offset += (2 * self.max_possible_edges) + (2 * self.max_node_count)

        # Calculate size of one iteration
        iteration_data_size = (
            self.image_shape +  # Current image
            2 +  # Brick coordinates
            1 +  # Reward
            (2 * self.max_possible_edges) +  # Edge indices
            (2 * self.max_node_count)  # Node values
        )

        # Skip to current step
        data_offset += step_idx * iteration_data_size

        # Get current image
        current_image = np.array(np.reshape(row_data[data_offset:data_offset + self.image_shape], (28, 28)))
        processed_current_image = self.preprocess_image(target_image)

        data_offset += self.image_shape

        # Get brick coordinates
        brick_coordinates = torch.tensor(
            row_data[data_offset:data_offset + 2],
            dtype=torch.float32
        )
        data_offset += 2

        # Get reward
        reward = row_data[data_offset]
        data_offset += 1

        # Get current graph data
        edge_indices = row_data[data_offset:data_offset + 2 * self.max_possible_edges]
        data_offset += 2*self.max_possible_edges
        
        node_values = row_data[data_offset:data_offset + 2 * self.max_node_count]
        
        # Convert to PyG Data object
        current_graph = self._process_graph_data(edge_indices, node_values)

        return target_image, processed_target_image, current_image, processed_current_image, brick_coordinates, reward, current_graph