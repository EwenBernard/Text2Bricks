from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data
import os
from typing import Tuple
import threading
from queue import Queue


class LegoPretrainDataset(Dataset):
    def __init__(self, dataset_dir: str, sample_size: int = 3000, cache_size: int = 5) -> None:
        """
        Args:
            dataset_dir (str): Path to the directory containing the dataset files.
            sample_size (int): Number of samples to load references for.
            cache_size (int): Number of files to cache in memory.
        """
        self.dataset_dir = dataset_dir
        self.sample_references = []
        self.cache = {}  # Caching dictionary
        self.cache_lock = threading.Lock()  # Lock for thread-safe cache access
        self.cache_size = cache_size
        self.cache_queue = Queue()
        self.preloaded_condition = threading.Condition(self.cache_lock)  # Synchronization for preloading

        # Index files and episodes without fully loading them
        self._index_dataset(sample_size)

        # Start the preloading thread
        self._stop_event = threading.Event()
        self.preloader_thread = threading.Thread(target=self._preload_files)
        self.preloader_thread.daemon = True
        self.preloader_thread.start()
        

    def _index_dataset(self, sample_size) -> None:
        """
        Index dataset files and episodes without fully loading them.
        """
        sample_counter = 0
        for file_name in os.listdir(self.dataset_dir):
            if file_name.endswith('.pt'):
                file_path = os.path.join(self.dataset_dir, file_name)
                data = torch.load(file_path, map_location='cpu')  # Load file metadata only
                
                for episode_idx in range(len(data["iteration_data"])):
                    self.sample_references.append((file_path, episode_idx))
                    sample_counter += 1
                    if sample_counter >= sample_size:
                        break

    def __len__(self) -> int:
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.sample_references)

    def _preload_files(self):
        """
        Preload files into the cache based on the next indices to be accessed.
        """
        while not self._stop_event.is_set():
            while not self.cache_queue.empty():
                idx = self.cache_queue.get()
                file_path, _ = self.sample_references[idx]

                # Check if the file is already cached
                with self.cache_lock:
                    if file_path not in self.cache:
                        # Load the file and add it to the cache
                        data = torch.load(file_path, map_location="cpu")
                        self.cache[file_path] = data

                        # Notify all waiting threads that this file is now cached
                        self.preloaded_condition.notify_all()

                        # Maintain cache size
                        if len(self.cache) > self.cache_size:
                            # Remove the oldest cached item
                            oldest_file = next(iter(self.cache))
                            del self.cache[oldest_file]


    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Data]:
        """
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing:
                - target_image (Tensor): Target image of the final structure.
                - current_build_image (Tensor): Current image of the structure being built.
                - brick_to_add (Tensor): Coordinates of the next brick to add.
                - reward (Tensor): Reward value for the action.
                - current_graph (torch_geometric.Data): Graph representation of the current state.
        """
        file_path, episode_idx = self.sample_references[idx]

        # Ensure the file is preloaded
        with self.cache_lock:
            while file_path not in self.cache:
                self.cache_queue.put(idx)  # Ensure the file is queued for preloading
                self.preloaded_condition.wait()  # Wait until the file is preloaded

            data = self.cache[file_path]

        # Extract the required episode data
        target_image = data["initial_data"]["target_image"]
        episode = data["iteration_data"][episode_idx]
        current_build_image = episode[0]
        brick_to_add = episode[1]
        reward = episode[2]
        current_graph = episode[3]
        current_graph.x = current_graph.x.float()
        current_graph.edge_index = current_graph.edge_index.long()

        # Add the next indices to the preload queue
        for i in range(1, self.cache_size + 1):
            next_idx = idx + i
            if next_idx < len(self.sample_references):
                self.cache_queue.put(next_idx)

        return (target_image, current_build_image, brick_to_add, reward, current_graph)

    def stop_preloader(self):
        """
        Stop the preloader thread.
        """
        self._stop_event.set()
        with self.cache_lock:
            self.preloaded_condition.notify_all()
        self.preloader_thread.join()