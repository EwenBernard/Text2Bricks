import os
import torch
from text2brick.models import GraphLegoWorldData
from text2brick.dataset import MNISTDataset
from .Preprocessing import PreprocessImage

class LegoDatasetGenerator:
    def __init__(self, output_dir: str = "./lego_dataset"):
        self.mnist = MNISTDataset()
        self.preprocess_image = PreprocessImage()
        self.output_dir = output_dir

    def generate(self, idx: int):
        array, _, _, _ = self.mnist.sample(sample_index=idx)
        lego_world = GraphLegoWorldData(array)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        iteration_data = []

        initial_data = {
            "target_image": self.preprocess_image(array),
            "initial_graph": lego_world.graph_to_torch(),
        }

        for _ in range(lego_world.nodes_num()):
           
            brick_to_remove = lego_world.get_brick_at_edge()
            lego_world.remove_brick(brick_to_remove.get("x"), brick_to_remove.get("y"))
            current_image = lego_world.graph_to_table()

            iteration_data.append([
                self.preprocess_image(current_image),  # Current image as a tensor
                brick_to_remove,  # Brick to remove
                lego_world.graph_to_torch(),  # Current graph
            ])

        torch.save({
            "initial_data": initial_data,
            "iteration_data": iteration_data,
        }, os.path.join(self.output_dir, f"{idx}.pt"))
                                              
