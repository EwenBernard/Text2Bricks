import os
import numpy as np
import h5py
from tqdm import tqdm
from typing import Tuple
import random

from text2brick.models import GraphLegoWorldData
from text2brick.gym import AbstractRewardFunc
from text2brick.dataset.MNISTDataset import MNISTDataset

class LegoDatasetGeneratorHDF5:
    def __init__(
        self, 
        output_file: str = "./lego_dataset/lego_dataset.h5",
        reward_function: AbstractRewardFunc = None,
        num_samples: int = 1000,
        image_shape: Tuple[int, int] = (28, 28),
        min_node_count: int = 40,
        random_next_node_frequency: float = 0.3
    ):
        self.mnist = MNISTDataset()
        self.output_file = output_file
        self.reward_function = reward_function
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.min_node_count = min_node_count
        self.random_next_node_frequency = random_next_node_frequency
        self.iteration_count = 0 

    def _initialize_hdf5(self):
        """
        Creates an HDF5 file and initializes datasets.
        """
        if not os.path.exists(os.path.dirname(self.output_file)):
            os.makedirs(os.path.dirname(self.output_file))

        self.h5_file = h5py.File(self.output_file, "w")
        self.h5_file.create_dataset(
            "target_images", 
            (self.num_samples, self.image_shape[0], self.image_shape[1]), 
            dtype=np.float32
        )
        self.h5_file.create_group("graphs")  # Store graph representations
        self.h5_file.create_group("iterations")  # Store iteration-specific data
        self.h5_file.create_dataset(
            "index_table", 
            (self.num_samples, 2), 
            dtype=np.int32
        )
        self.h5_file.attrs["num_samples"] = self.num_samples

    def _process_graph(self, lego_world: GraphLegoWorldData):
        data = lego_world.graph_to_torch()
        return data.edge_index.numpy(), data.x.numpy()
    

    def generate_sample(self, mnist_idx: int, idx: int):
        """
        Generate a single sample and store it in the HDF5 file.
        """
        array, _, _ = self.mnist.sample(sample_index=mnist_idx)
        lego_world = GraphLegoWorldData(array)

        if lego_world.nodes_num() < self.min_node_count:
            return False

        self.h5_file["target_images"][idx] = array

        # Store initial graph data
        edges, node_values = self._process_graph(lego_world)
        graph_group = self.h5_file["graphs"].create_group(f"sample_{idx}")
        graph_group.create_dataset("edges", data=edges, dtype=np.int32)
        graph_group.create_dataset("node_values", data=node_values, dtype=np.int32)

        global_starting_index = self.iteration_count
        num_iterations = lego_world.nodes_num()
        reward = 0
        self.reward_function.last_iou = 0
        i = 0
        random_count = 0

        # Process iterations
        iteration_group = self.h5_file["iterations"].create_group(f"sample_{idx}")

        while i < num_iterations:
            if random.random() < self.random_next_node_frequency: # Invalid brick
                validity = False
                brick_to_remove = lego_world.get_random_invalid_position()
                i -= 1
                random_count += 1
                # TODO: Valid but decrease iou growth
            else: # Valid brick
                brick_to_remove = lego_world.get_brick_at_edge()
                lego_world.remove_brick(brick_to_remove.get("x"), brick_to_remove.get("y"), debug=False)
                validity = True

            # Generate current image and compute reward
            current_image = lego_world.graph_to_table()
            reward += self.reward_function(array, current_image, validity)[0]

            # Process current state
            edges, node_values = self._process_graph(lego_world)

            # Store iteration data
            iter_data = iteration_group.create_group(f"iteration_{i + random_count}")
            iter_data.create_dataset("current_image", data=current_image, dtype=np.float32)
            iter_data.create_dataset(
                "brick_to_remove", 
                data=[brick_to_remove.get("x"), brick_to_remove.get("y")], 
                dtype=np.int32
            )
            iter_data.create_dataset("reward", data=reward, dtype=np.float32)
            iter_data.create_dataset("edges", data=edges, dtype=np.int32)
            iter_data.create_dataset("node_values", data=node_values, dtype=np.float32)

            i += 1

        # Update iteration count and index table
        global_ending_index = global_starting_index + num_iterations
        self.iteration_count = global_ending_index
        self.h5_file["index_table"][idx] = [global_starting_index, global_ending_index]

        return True

    def generate_dataset(self):
        """
        Generates the entire dataset and writes it to the HDF5 file.
        """
        print("Initializing HDF5 File")
        self._initialize_hdf5()

        generated_samples = 0
        mnist_idx = 0
        with tqdm(total=self.num_samples, desc="Generating dataset") as pbar:
            while generated_samples < self.num_samples:
                if self.generate_sample(mnist_idx, generated_samples):
                    generated_samples += 1
                    pbar.update(1)  # Update progress bar only on success
                if generated_samples >= self.num_samples:
                    break
                mnist_idx += 1

        self.h5_file.attrs["num_iterations"] = self.iteration_count

        print("Closing HDF5 File")
        self.h5_file.close()
        print("Done!")