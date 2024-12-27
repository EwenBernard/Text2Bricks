from torch.utils.data import Dataset
import torch
from torch_geometric.data import Batch, Data
import os

class LegoPretrainDataset(Dataset):
    def __init__(self, dataset_dir: str, sample_size: int = 50):
        """
        Args:
            dataset_dir (str): Path to the directory containing the dataset files.
        """
        self.dataset_dir = dataset_dir
        self.samples = []

        # Load and flatten all episodes
        self._load_and_flatten_dataset(sample_size=sample_size)

    def _load_and_flatten_dataset(self, sample_size=50):
        """
        Load dataset files and flatten episodes into independent samples.
        """
        sample_counter = 0
        for file_name in os.listdir(self.dataset_dir):
            file_path = os.path.join(self.dataset_dir, file_name)
            if file_name.endswith('.pt'):
                data = torch.load(file_path)

                target_image = data["initial_data"]["target_image"]
                initial_graph = data["initial_data"]["initial_graph"]

                for episode in data["iteration_data"]:
                    current_graph = episode[3]
                    current_graph.x = current_graph.x.float()
                    current_graph.edge_index = current_graph.edge_index.long()

                    self.samples.append({
                        "target_image": target_image,
                        "initial_graph": initial_graph,
                        "current_build_image": episode[0],
                        "brick_to_add": episode[1],
                        "reward": episode[2],
                        "current_graph": current_graph,
                    })
                sample_counter += 1
                if sample_counter >= sample_size:
                    break

    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing:
                - target_image (Tensor): Target image of the final structure.
                - current_build_image (Tensor): Current image of the structure being built.
                - brick_to_add (dict): Coordinates of the next brick to add.
                - reward (float): Reward value for the action.
                - current_graph (torch_geometric.Data or custom): Graph representation of the current state.
        """
        sample = self.samples[idx]
        return (sample["target_image"], sample["current_build_image"], sample["brick_to_add"], sample["reward"], sample["current_graph"])