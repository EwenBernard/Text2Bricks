import torch
from torch.utils.data import Dataset
import numpy as np
import random

from text2brick.models import GraphLegoWorldData
from text2brick.dataset.dataset import MNISTDataset


class CustomDatasetGraph(Dataset):
    
    def __init__(self):
        self.mnist = MNISTDataset()
        self.length = 1

    def __len__(self):
        # Return the number of samples
        return self.length

    def __getitem__(self, index):

        array, _, _, _ = self.mnist.sample()
        mapping_table = [[0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 1, 1]]
  
        lego_world_ref = GraphLegoWorldData(table=mapping_table,)

        # TODO: Generate random index between 0 and len(graphs)
        random_index = random.randint(0, lego_world_ref.nodes_num())
        # TODO: Remove random_index nodes and clear the conections to get a new graphs

        # TODO: Return the new graph and the last node removed