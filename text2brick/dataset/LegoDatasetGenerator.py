import os
import torch
from text2brick.models import GraphLegoWorldData
from text2brick.dataset import MNISTDataset
from .Preprocessing import PreprocessImage
from tqdm import tqdm

class LegoDatasetGenerator:
    def __init__(self, output_dir: str = "./lego_dataset"):
        self.mnist = MNISTDataset()
        self.preprocess_image = PreprocessImage()
        self.output_dir = output_dir

    def generate(self, idx: int, save_iteration_graph=True):
        array, _, _, _ = self.mnist.sample(sample_index=idx)
        lego_world = GraphLegoWorldData(array)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        iteration_data = []

        initial_data = {
            "target_image": self.preprocess_image(array),
            "initial_graph": lego_world.graph_to_torch(deepcopy=True),
        }

        for _ in range(lego_world.nodes_num()):
           
            brick_to_remove = lego_world.get_brick_at_edge()
            lego_world.remove_brick(brick_to_remove.get("x"), brick_to_remove.get("y"))
            current_image = lego_world.graph_to_table()

            if save_iteration_graph:
                iteration_data.append([
                    self.preprocess_image(current_image),  # Current image as a tensor
                    brick_to_remove,  # Brick to remove
                    lego_world.graph_to_torch(deepcopy=True),  # Current graph
                ])
            else: 
                iteration_data.append([
                    self.preprocess_image(current_image),  # Current image as a tensor
                    brick_to_remove,  # Brick to remove
                ])

        torch.save({
            "initial_data": initial_data,
            "iteration_data": iteration_data,
        }, os.path.join(self.output_dir, f"{idx}.pt"))
                                              

    def generate_dataset(self, num_samples: int, save_iteration_graph=True):
        for idx in tqdm(range(num_samples)):
            self.generate(idx, save_iteration_graph=save_iteration_graph)