from torch.utils.data import Dataset
import torch
from torch_geometric.data import Batch, Data
import os

class LegoPretrainDataset(Dataset):
    def __init__(self, dataset_dir: str):
        """
        Args:
            dataset_dir (str): Path to the directory containing the dataset files.
        """
        self.dataset_dir = dataset_dir
        self.samples = []

        # Load and flatten all episodes
        self._load_and_flatten_dataset()

    def _load_and_flatten_dataset(self):
        """
        Load dataset files and flatten episodes into independent samples.
        """
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

                    if current_graph.num_nodes == 0:
                        # Create a valid empty graph
                        #TODO need to modify the dtype when changing the coord system to non int coords
                        current_graph = Data(
                            x = torch.empty(0, 2, dtype=torch.float),
                            edge_index = torch.empty(2, 0, dtype=torch.long),  # No edges
                            validity = torch.empty(0)
                        )
                    
                    self.samples.append({
                        "target_image": target_image,
                        "initial_graph": initial_graph,
                        "current_build_image": episode[0],
                        "brick_to_add": episode[1],
                        "reward": episode[2],
                        "current_graph": current_graph,
                    })

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
        #return sample["current_graph"]

    # def custom_collate_fn(batch):
    #     try:
    #         # Stack fixed-size tensors
    #         target_images = torch.stack([item[0] for item in batch])
    #         current_build_images = torch.stack([item[1] for item in batch])
    #         bricks_to_add = torch.stack([item[2] for item in batch])  # Shape [batch_size, 2]
    #         rewards = torch.tensor([item[3].item() for item in batch], dtype=torch.float32)

    #         # Batch graph data
    #         current_graphs = Batch.from_data_list([item[4] for item in batch])

    #         return (target_images, current_build_images, bricks_to_add, rewards, current_graphs)
                
    #     except Exception as e:
    #         print("Error in custom collate function:")
    #         print(f"Batch: {batch}")
    #         raise e