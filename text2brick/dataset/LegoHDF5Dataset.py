import os
import psutil
import torch
from torch.utils.data import Dataset
import h5py
from torch_geometric.data import Data
from typing import Tuple
import numpy as np

from text2brick.dataset.Preprocessing import PreprocessImage

class LegoHDF5Dataset(Dataset):
    def __init__(
        self, 
        hdf5_file: str = "./lego_dataset/lego_dataset.h5",
    ) -> None:
        """
        Args:
            hdf5_file (str): Path to the HDF5 dataset file.
        """
        self.hdf5_file = hdf5_file
        self.processing = PreprocessImage()

        self._init_dataset()

        self.total_iterations = self.index_table[-1, 1]  # Total iterations across all samples

    def _init_dataset(self):
        """
        Initialize the dataset by loading the entire dataset into memory or using lazy loading.
        """
        # Get the size of the HDF5 file in memory
        h5_file_size = os.path.getsize(self.hdf5_file)
        available_memory = psutil.virtual_memory().available
        print(f"Available memory: {available_memory / 1024**3:.2f} GB")
        print(f"HDF5 file size: {h5_file_size / 1024**3:.2f} GB")

        if h5_file_size <= 0.5 * available_memory:
            print("Loading entire dataset into memory...")
            self.in_memory = True
            with h5py.File(self.hdf5_file, "r") as h5_file:
                self.data = {
                    "target_images": h5_file["target_images"][:],
                    "index_table": h5_file["index_table"][:],
                    "iterations": {
                        key: {
                            iter_key: {
                                "current_image": iter_group["current_image"][:],
                                "brick_to_remove": iter_group["brick_to_remove"][:],
                                "reward": iter_group["reward"][()],
                                "edges": iter_group["edges"][:],
                                "node_values": iter_group["node_values"][:],
                            } for iter_key, iter_group in h5_file["iterations"][key].items()
                        } for key in h5_file["iterations"].keys()
                    }
                }
        else:
            print("Using lazy loading for dataset...")
            self.in_memory = False
            self.h5_file = h5py.File(self.hdf5_file, "r")

        if self.in_memory:
            self.num_samples = len(self.data["target_images"])
            self.index_table = self.data["index_table"]
        else:
            self.num_samples = len(self.h5_file["target_images"])
            self.index_table = self.h5_file["index_table"][:]


    def __len__(self) -> int:
        """
        Returns:
            int: Total number of iterations in the dataset.
        """
        return self.total_iterations

    def _find_sample_and_iter(self, idx: int) -> Tuple[int, int]:
        """
        Find the sample index and iteration index for a given dataset index.

        Args:
            idx (int): Global iteration index.

        Returns:
            tuple: (sample_idx, iter_idx) where:
                - sample_idx: The index of the sample.
                - iter_idx: The iteration index within the sample.
        """
        sample_idx = np.searchsorted(self.index_table[:, 1], idx, side="right")
        sample_start = self.index_table[sample_idx, 0]
        iter_idx = idx - sample_start
        return sample_idx, iter_idx

    def _process_graph_data(self, edges, node_values) -> Data:
        """
        Convert edge indices and node values to PyG Data object.

        Args:
            edges (np.ndarray): Array of edge indices (shape: [2, num_edges]).
            node_values (np.ndarray): Array of node features.

        Returns:
            Data: PyTorch Geometric Data object.
        """
        if len(node_values) == 0:
            return Data(
                x=torch.empty((0, 2), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long)
            )

        edge_index = torch.tensor(edges, dtype=torch.long)  # Convert to [2, num_edges]
        node_features = torch.tensor(node_values, dtype=torch.float32)

        return Data(x=node_features, edge_index=edge_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Data, dict]:
        """
        Args:
            idx (int): Index of the iteration to retrieve.

        Returns:
            tuple: (target_image, initial_graph, iteration_data)
        """
        sample_idx, iter_idx = self._find_sample_and_iter(idx)

        if self.in_memory:
            target_image = self.data["target_images"][sample_idx]
            iteration_group = self.data["iterations"][f"sample_{sample_idx}"][f"iteration_{iter_idx}"]
        else:
            target_image = self.h5_file["target_images"][sample_idx]
            iteration_group = self.h5_file["iterations"][f"sample_{sample_idx}"][f"iteration_{iter_idx}"]

        current_image = iteration_group["current_image"][:]
        brick_to_remove = torch.tensor(iteration_group["brick_to_remove"], dtype=torch.long)
        reward = torch.tensor(iteration_group["reward"], dtype=torch.float32)

        edges = iteration_group["edges"][:]
        node_values = iteration_group["node_values"][:]
        current_graph = self._process_graph_data(edges, node_values)

        # Process images
        processed_target_image = self.processing(target_image)
        processed_current_image = self.processing(current_image)
        target_image = torch.tensor(target_image, dtype=torch.float32)
        current_image = torch.tensor(current_image, dtype=torch.float32)

        return target_image, current_image, processed_target_image, processed_current_image, brick_to_remove, reward, current_graph