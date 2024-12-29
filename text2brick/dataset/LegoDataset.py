from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data
import os
from typing import Tuple
from tqdm import tqdm


class LegoPretrainDataset(Dataset):
    def __init__(self, dataset_dir: str, sample_size: int = 3000) -> None:
        """
        Args:
            dataset_dir (str): Path to the directory containing the dataset files.
        """
        self.dataset_dir = dataset_dir
        self.sample_references = []

        # Index files and episodes without fully loading them
        self._index_dataset(sample_size)
        self.tqdm_bar = tqdm(total=len(self.sample_references), desc="Loading samples", position=0, leave=True)

    def _index_dataset(self, sample_size) -> None:
        """
        Index dataset files and episodes without fully loading them.
        """
        sample_counter = 0
        for file_name in os.listdir(self.dataset_dir):
            if file_name.endswith('.pt'):
                file_path = os.path.join(self.dataset_dir, file_name)
                data = torch.load(file_path, map_location='cpu')  # Load file metadata only
                
                for episode_idx in range(len(data["iteration_data"])):
                    self.sample_references.append((file_path, episode_idx))
                    sample_counter += 1
                    if sample_counter >= sample_size:
                        break

    def __len__(self) -> int:
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.sample_references)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Data]:
        """
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing:
                - target_image (Tensor): Target image of the final structure.
                - current_build_image (Tensor): Current image of the structure being built.
                - brick_to_add (Tensor): Coordinates of the next brick to add.
                - reward (Tensor): Reward value for the action.
                - current_graph (torch_geometric.Data): Graph representation of the current state.
        """
        self.tqdm_bar.update(1)

        file_path, episode_idx = self.sample_references[idx]
        data = torch.load(file_path, map_location='cpu')  # Load file lazily

        target_image = data["initial_data"]["target_image"]
        episode = data["iteration_data"][episode_idx]
        current_build_image = episode[0]
        brick_to_add = episode[1]
        reward = episode[2]

        current_graph = episode[3]
        current_graph.x = current_graph.x.float()
        current_graph.edge_index = current_graph.edge_index.long()

        return (target_image, current_build_image, brick_to_add, reward, current_graph)