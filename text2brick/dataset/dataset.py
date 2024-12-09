from PIL import Image
import numpy as np
from sklearn.datasets import fetch_openml
import logging

from text2brick.utils.ImageUtils import array_to_image


class Dataset:
    """
    A class to handle MNIST dataset operations, including fetching data, sampling random images, 
    and modifying samples by truncating rows or columns.
    """

    def __init__(self):
        """
        Initializes the Dataset by loading the MNIST data from OpenML.
        """
        mnist = fetch_openml('mnist_784', version=1)  # Load MNIST dataset

        self.data = mnist.data  # 70000 samples, each with 784 features (28x28 pixels flattened)
        self.labels = mnist.target  # Corresponding labels for the dataset


    def _random_sample(self):
        """
        Generates a random sample (image) from the MNIST dataset.

        Returns:
            np.ndarray: A 28x28 binary array representing the sampled image.
        """
        random_index = np.random.randint(0, self.data.shape[0])  # Random index from the dataset
        image_array = self.data.iloc[random_index].to_numpy().reshape(28, 28)  # Reshape to 28x28 pixels
        image_label = self.labels[random_index]

        # Convert the grayscale image into binary (0 or 1)
        image_array = np.where(image_array > 0, 1, 0).astype(np.uint8)

        return image_array


    def sample(self):
        """
        Fetches a random sample from the dataset and converts it into an image.

        Returns:
            tuple: A tuple containing the binary array (28x28) and its corresponding image object.
        """
        sample_array = self._random_sample()
        sample_image = array_to_image(sample_array)

        return sample_array, sample_image


    def sample_truncated_horizontally(self, n_rows: int):
        """
        Fetches a random sample and truncates its top `n_rows` by setting them to 0.

        Args:
            n_rows (int): Number of rows to truncate.

        Raises:
            ValueError: If `n_rows` exceeds the total number of rows in the image.

        Returns:
            tuple: A tuple containing the modified binary array and its corresponding image object.
        """
        sample_array = self._random_sample()

        if n_rows > sample_array.shape[0]:
            raise ValueError(
                f"Requested number of rows ({n_rows}) exceeds the available rows ({sample_array.shape[0]})."
            )

        sample_array[:n_rows, :] = 0
        sample_image = array_to_image(sample_array)

        return sample_array, sample_image


    def sample_truncated_vertically(self, n_columns: int):
        """
        Fetches a random sample and truncates its leftmost `n_columns` by setting them to 0.

        Args:
            n_columns (int): Number of columns to truncate.

        Raises:
            ValueError: If `n_columns` exceeds the total number of columns in the image.

        Returns:
            tuple: A tuple containing the modified binary array and its corresponding image object.
        """
        sample_array = self._random_sample()

        if n_columns > sample_array.shape[1]:
            raise ValueError(
                f"Requested number of columns ({n_columns}) exceeds the available columns ({sample_array.shape[1]})."
            )

        sample_array[:, :n_columns] = 0
        sample_image = array_to_image(sample_array)

        return sample_array, sample_image