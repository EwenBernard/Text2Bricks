from text2brick.models import GraphLegoWorldData
from text2brick.gym import AbstractRewardFunc
from text2brick.dataset.MNISTDataset import MNISTDataset
from text2brick.dataset.Preprocessing import PreprocessImage

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from tqdm import tqdm
import os
import torch


class LegoDatasetGenerator:
    def __init__(self, output_dir: str = "./lego_dataset", reward_function: AbstractRewardFunc = None, batch_size: int = 500, num_workers: int = 4):
        self.mnist = MNISTDataset()
        self.preprocess_image = PreprocessImage()
        self.output_dir = output_dir
        self.reward_function = reward_function
        self.batch_size = batch_size
        self.num_workers = num_workers

    def generate(self, idx: int, progress_counter=None, save_iteration_graph=True):
        """
        Generate a single sample and update the shared progress counter.

        Args:
            idx (int): Index of the sample.
            progress_counter (multiprocessing.Value): Shared counter for progress tracking.
            save_iteration_graph (bool): Whether to save iteration graph data.

        Returns:
            dict: The generated data for the sample.
        """
        array, _, _ = self.mnist.sample(sample_index=idx)
        lego_world = GraphLegoWorldData(array)

        initial_data = {
            "target_image": self.preprocess_image(array),
            "initial_graph": lego_world.graph_to_torch(deepcopy=True),
        }

        iteration_data = []

        for _ in range(lego_world.nodes_num()):
            brick_to_remove = lego_world.get_brick_at_edge()
            lego_world.remove_brick(brick_to_remove.get("x"), brick_to_remove.get("y"))
            current_image = lego_world.graph_to_table()

            reward = self.reward_function(array, current_image, 1)

            iteration_entry = {
                "current_image": self.preprocess_image(current_image),
                "next_brick": torch.tensor([brick_to_remove.get("x"), brick_to_remove.get("y")]),
                "reward": torch.tensor(reward),
            }
            
            if save_iteration_graph:
                iteration_entry["current_graph"] = lego_world.graph_to_torch(deepcopy=True)
            
            iteration_data.append(iteration_entry)

            # Release memory after use
            del brick_to_remove, current_image

        # Update shared progress counter
        if progress_counter is not None:
            progress_counter.value += 1

        return {
            "initial_data": initial_data,
            "iteration_data": iteration_data,
        }

    def generate_batch(self, indices, progress_counter, save_iteration_graph):
        """
        Generates a batch of episodes for the given indices, updating the shared progress counter.
        """
        batch_data = []
        for idx in indices:
            batch_data.append(self.generate(idx, progress_counter=progress_counter, save_iteration_graph=save_iteration_graph))
        
        # Release memory after batch is generated
        del indices  # Free memory for batch indices
        return batch_data

    def generate_dataset(self, num_samples: int, save_iteration_graph=True):
        """
        Generate the entire dataset with a shared progress counter.

        Args:
            num_samples (int): Number of samples to generate.
            save_iteration_graph (bool): Whether to save iteration graph data.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        indices = list(range(num_samples))
        batch_indices = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        with Manager() as manager:
            # Shared counter for progress tracking
            progress_counter = manager.Value('i', 0)

            # Progress bar watching the shared counter
            with tqdm(total=num_samples, desc="Generating dataset", position=0) as pbar:
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [
                        executor.submit(self.generate_batch, batch, progress_counter, save_iteration_graph)
                        for batch in batch_indices
                    ]
                    # Update the progress bar based on the shared counter
                    while any(future.running() for future in futures):
                        pbar.n = progress_counter.value
                        pbar.last_print_n = progress_counter.value  # Forces update of progress bar
                        pbar.refresh()

                    # Ensure all processes complete
                    for future in futures:
                        batch_data = future.result()
                        batch_file = os.path.join(self.output_dir, f"batch_{futures.index(future)}.pt")
                        torch.save(batch_data, batch_file)
                        del batch_data  # Free memory after saving to disk
