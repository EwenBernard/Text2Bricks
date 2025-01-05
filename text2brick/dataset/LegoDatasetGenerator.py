from text2brick.models import GraphLegoWorldData
from text2brick.gym import AbstractRewardFunc
from text2brick.dataset.MNISTDataset import MNISTDataset
from text2brick.dataset.Preprocessing import PreprocessImage

import os
import numpy as np
from tqdm import tqdm


class LegoDatasetGenerator:
    def __init__(
        self, 
        output_file: str = "./lego_dataset/lego_dataset.dat",
        reward_function: AbstractRewardFunc = None,
        num_samples: int = 1000,
        image_shape: int = 150528,
        max_node_count: int = 140,
    ):
        self.mnist = MNISTDataset()
        self.preprocess_image = PreprocessImage()
        self.output_file = output_file
        self.reward_function = reward_function
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.max_node_count = max_node_count

        # Define memmap file layout
        self.total_columns = (
            image_shape +  # Target image
            pow(max_node_count, 2) +  # Initial node tensor
            max_node_count * (  # Iteration data
                image_shape +  # Current image
                2 +  # Brick coordinates (x, y)
                1 +  # Reward
                pow(max_node_count, 2)
            )
        )

    def _initialize_memmap(self):
        """
        Creates a memmap file and allocates space for the dataset.
        """
        if not os.path.exists(os.path.dirname(self.output_file)):
            os.makedirs(os.path.dirname(self.output_file))
        
        # Create memmap with shape (num_samples, total_columns)
        self.data_memmap = np.memmap(
            self.output_file, dtype=np.float32, mode="w+",
            shape=(self.num_samples, self.total_columns)
        )

    def _process_graph(self, lego_world: GraphLegoWorldData):
        graph_array = lego_world.graph_to_np().flatten()
        max_graph_size = pow(self.max_node_count, 2)
        return np.pad(graph_array, (0, max_graph_size - len(graph_array)), constant_values=0)
    

    def generate_sample(self, idx):
        """
        Optimized sample generation with pre-allocated NumPy array and avoiding conversion.
        """
        #TODO PROBLEM ON HOW TO REMOVE THE PADDING - NODE VALUES ARE NOT STORED IN THE MEMMAP - KEEP TENSOR LIKE APPROACH WITH SEPARATE NODES AND EDGES
        array, _, _ = self.mnist.sample(sample_index=idx)
        lego_world = GraphLegoWorldData(array)

        target_image = self.preprocess_image(array).flatten()
        graph_array = self._process_graph(lego_world)

        # Pre-allocate the row_data array to the max possible size of a row
        max_size = self.total_columns  # Assuming max_size is already known (i.e., total_columns)
        row_data = np.zeros(max_size, dtype=np.float32)

        # Place target image and graph into the row_data
        row_data[:len(target_image)] = target_image
        row_data[len(target_image):len(target_image) + len(graph_array)] = graph_array

        data_offset = len(target_image) + len(graph_array)  # Offset to start placing iteration data
        for i in range(lego_world.nodes_num()):
            if i >= self.max_node_count:
                break

            # Get and remove brick
            brick_to_remove = lego_world.get_brick_at_edge()
            lego_world.remove_brick(brick_to_remove.get("x"), brick_to_remove.get("y"))

            # Generate current image and compute reward
            current_image = lego_world.graph_to_table()
            reward = self.reward_function(array, current_image, 1)

            # Preprocess image and graph
            current_image = self.preprocess_image(current_image).flatten()
            current_graph_array = self._process_graph(lego_world)

            # Place iteration data into the row_data array
            row_data[data_offset:data_offset + len(current_image)] = current_image
            data_offset += len(current_image)
            row_data[data_offset:data_offset + 1] = brick_to_remove.get("x")
            data_offset += 1
            row_data[data_offset:data_offset + 1] = brick_to_remove.get("y")
            data_offset += 1
            row_data[data_offset:data_offset + 1] = reward
            data_offset += 1
            row_data[data_offset:data_offset + len(current_graph_array)] = current_graph_array
            data_offset += len(current_graph_array)

        # Ensure the row_data has the correct length by padding if necessary
        if data_offset < max_size:
            row_data[data_offset:] = 0  # Pad the remaining space with zeros (no need to do np.pad)

        self.data_memmap[idx, :] = row_data


    def generate_dataset(self):
        """
        Generates the entire dataset and writes it to the memmap file.
        """
        print("Memmap database init")
        self._initialize_memmap()

        indices = list(range(self.num_samples))
        for idx in tqdm(indices, desc="Generating dataset"):
            self.generate_sample(idx)

        print("Saving memmap database")
        self.data_memmap.flush()