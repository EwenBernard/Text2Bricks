import os
import torch
from torch.utils.data import Dataset
import networkx as nx
from typing import List, Tuple
from text2brick.models import GraphLegoWorldData
from text2brick.dataset import MNISTDataset

class LegoDatasetGenerator: 
    def __init__(self, output_dir: str = "./lego_dataset"):
        self.mnist = MNISTDataset()
        self.output_dir = output_dir

    def generate(self, idx: int):
        array, _, _, _ = self.mnist.sample(sample_index=idx)
        lego_world = GraphLegoWorldData(array)

        idx_dir = os.path.join(self.output_dir, f"{idx}")
        if not os.path.exists(idx_dir):
            os.makedirs(idx_dir)

        self.save_initial_data(array, lego_world, idx_dir)
        
        for i in range(lego_world.nodes_num()):
            
            brick_to_remove = lego_world.get_brick_at_edge()
            current_graph = lego_world.graph.copy()
            current_graph = lego_world.remove_brick(brick_to_remove.get("x"), brick_to_remove.get("y"))
            current_image = lego_world.graph_to_table()

            torch.save({
                "iteration": i,
                "current_image": current_image,
                "brick_to_remove": brick_to_remove,
                "current_graph": current_graph,
            }, os.path.join(idx_dir, f"step_{i}.pt"))

    def save_initial_data(self, array, lego_world, idx_dir):
        torch.save({
            "target_image": array,
            "initial_graph": lego_world.graph,
        }, os.path.join(idx_dir, "initial_data.pt"))