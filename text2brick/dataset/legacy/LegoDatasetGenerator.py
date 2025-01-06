from text2brick.models import GraphLegoWorldData
from text2brick.gym import AbstractRewardFunc
from text2brick.dataset.MNISTDataset import MNISTDataset

import os
import numpy as np
from tqdm import tqdm


PADDING_VALUE = -1


class LegoDatasetGenerator:
    def __init__(
        self, 
        output_file: str = "./lego_dataset/lego_dataset.dat",
        reward_function: AbstractRewardFunc = None,
        num_samples: int = 1000,
        image_shape: int = 784,
        max_node_count: int = 140,
        min_node_count: int = 40
    ):
        self.mnist = MNISTDataset()
        self.output_file = output_file
        self.reward_function = reward_function
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.max_node_count = max_node_count
        self.min_node_count = min_node_count

        # Define memmap file layout with updated graph representation
        max_possible_edges = pow(max_node_count, 2)
        self.total_columns = (
            image_shape +  # Target image
            (2 * max_possible_edges) +  # Edge indices (flattened [num_edges, 2] array)
            max_node_count +  # Node values
            max_node_count * (  # Iteration data
                image_shape +  # Current image
                2 +  # Brick coordinates (x, y)
                1 +  # Reward
                (2 * max_possible_edges) +  # Edge indices for current state
                (2 * max_node_count)
            )
        )

    def _initialize_memmap(self):
        """
        Creates a memmap file and allocates space for the dataset.
        """
        if not os.path.exists(os.path.dirname(self.output_file)):
            os.makedirs(os.path.dirname(self.output_file))
        
        self.data_memmap = np.memmap(
            self.output_file, dtype=np.float32, mode="w+",
            shape=(self.num_samples, self.total_columns)
        )

    def _process_graph(self, lego_world: GraphLegoWorldData):
        # Get adjacency matrix and convert to edge_index format
        adj_matrix = lego_world.graph_to_np()
        edges = np.argwhere(adj_matrix == 1)  # Gets indices where value is 1
        num_edges = len(edges)
        
        # Calculate required padding
        max_possible_edges = self.max_node_count * self.max_node_count
        padding_size = max_possible_edges - num_edges
        
        # Pad edges with -1 if needed
        if padding_size > 0:
            padding = np.full((padding_size, 2), -1)
            edges = np.vstack([edges, padding])
        
        node_values = np.array([(data.get('x', -1), data.get('y', -1)) for _, data in lego_world.get_nodes()])

        # Pad node values to match `max_node_count` rows with 2 columns
        if len(node_values) < self.max_node_count:
            padding = np.full((self.max_node_count - len(node_values), 2), PADDING_VALUE)
            node_values = np.vstack([node_values, padding])
        
        # Flatten everything into a single array
        # Format: [edge_pairs_flattened, node_values]
        combined = np.concatenate([edges.flatten(), node_values.flatten()])
        
        return combined

    def generate_sample(self, mnist_idx: int, idx: int):
        """
        Generate sample with updated graph representation.
        """
        array, _, _ = self.mnist.sample(sample_index=mnist_idx)
        lego_world = GraphLegoWorldData(array)

        if lego_world.nodes_num() < self.min_node_count:
            return False

        target_image = array.flatten()
        graph_array = self._process_graph(lego_world)

        # Pre-allocate the row_data array
        row_data = np.full(self.total_columns, -1, dtype=np.float32) 

        # Place target image and initial graph data
        current_pos = 0
        
        # Store target image
        row_data[current_pos:current_pos + len(target_image)] = target_image
        current_pos += len(target_image)
        
        # Store initial graph data
        row_data[current_pos:current_pos + len(graph_array)] = graph_array
        current_pos += len(graph_array)

        # Process iterations
        for i in range(lego_world.nodes_num()):
            if i >= self.max_node_count:
                break

            # Get and remove brick
            brick_to_remove = lego_world.get_brick_at_edge()
            lego_world.remove_brick(brick_to_remove.get("x"), brick_to_remove.get("y"))

            # Generate current image and compute reward
            current_image = lego_world.graph_to_table()
            reward = self.reward_function(array, current_image, 1)

            # Process current state
            current_image = current_image.flatten()
            current_graph_array = self._process_graph(lego_world)

            # Store iteration data
            row_data[current_pos:current_pos + len(current_image)] = current_image
            current_pos += len(current_image)
            row_data[current_pos] = brick_to_remove.get("x")
            current_pos += 1
            row_data[current_pos] = brick_to_remove.get("y")
            current_pos += 1
            row_data[current_pos] = reward
            current_pos += 1
            row_data[current_pos:current_pos + len(current_graph_array)] = current_graph_array
            current_pos += len(current_graph_array)

        # Any remaining space is already zeroed from initialization

        self.data_memmap[idx, :] = row_data

        return True

    def generate_dataset(self):
        """
        Generates the entire dataset and writes it to the memmap file.
        """
        print("Init Memmap")
        self._initialize_memmap()

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


        print("Saving Memmap Data")
        self.data_memmap.flush()
        self.data_memmap._mmap.close()
        del self.data_memmap  # Remove the reference
        print("Done!")