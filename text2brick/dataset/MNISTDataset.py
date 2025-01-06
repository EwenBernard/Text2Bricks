from PIL import Image
import numpy as np
from sklearn.datasets import fetch_openml
from typing import Tuple
from joblib import Memory
from text2brick.utils.ImageUtils import array_to_image


class MNISTDataset:
    """
    A class to handle MNIST dataset operations, including fetching data, sampling random images, 
    and modifying samples by truncating rows or columns.
    """

    def __init__(self, cache_dir: str = './tmp', caching=True) -> None:
        """
        Initializes the Dataset by loading the MNIST data from OpenML.

        Args:
            cache_dir (str): Directory to cache the fetched data.
        """
        if caching:
            memory = Memory(cache_dir)
            fetch_openml_cached = memory.cache(fetch_openml)
            mnist = fetch_openml_cached('mnist_784', version=1)
        else:
            mnist = fetch_openml('mnist_784', version=1)

        self.data = mnist.data

    def sample(self, sample_index=None, n_cols=None, n_rows=None) -> Tuple[np.array, Image.Image, int]:
        """
        Fetches a sample from the dataset, modifies it by truncating rows or columns if specified, 
        and converts it into an image.

        Args:
            sample_index (int, optional): Index of the sample to fetch. Defaults to a random sample.
            n_cols (int, optional): Number of leftmost columns to truncate (set to zero). Defaults to None.
            n_rows (int, optional): Number of topmost rows to truncate (set to zero). Defaults to None.

        Returns:
            tuple: A tuple containing the binary array (28x28), its corresponding image object, and index.
        """
        array, index = self._sample_array(sample_index=sample_index)

        # Truncate specified number of columns by setting their values to 0
        if n_cols:
            if n_cols > array.shape[1]:
                raise ValueError(
                    f"Requested number of columns ({n_cols}) exceeds the available columns ({array.shape[1]})."
                )
            array[:, :n_cols] = 0

        # Truncate specified number of rows by setting their values to 0
        if n_rows:
            if n_rows > array.shape[0]:
                raise ValueError(
                    f"Requested number of rows ({n_rows}) exceeds the available rows ({array.shape[0]})."
                )
            array[:n_rows, :] = 0

        image = array_to_image(array)

        return array, image, index

    def _sample_array(self, sample_index=None):
        """
        Retrieves a single sample from the dataset and preprocesses it.

        Args:
            sample_index (int, optional): Index of the sample to retrieve. Defaults to a random index.

        Returns:
            tuple: The preprocessed binary array (28x28) and the sample index.
        """
        if sample_index is None:  # If no index is provided, pick a random sample
            sample_index = np.random.randint(0, self.data.shape[0])
        elif sample_index >= len(self.data):
            raise ValueError(
                f"Index requested {sample_index} is out of range, the length of the dataset is {len(self.data)}"
            )

        # Retrieve the flattened image
        image = self.data.iloc[sample_index]

        # Preprocess the image into binary 28x28 format
        image_array = self._preprocess_sample(image)

        return image_array, sample_index

    def _preprocess_sample(self, image):
        """
        Preprocesses a single sample by reshaping, reordering rows, and binarizing the pixel values.

        Args:
            image (pd.Series): Flattened image from the dataset.

        Returns:
            np.array: Binary 28x28 numpy array representation of the image.
        """
        # Reshape the image from flattened (784,) to 2D (28x28)
        image = image.to_numpy().reshape(28, 28)

        # Identify and separate all-zero rows and non-zero rows to place all-zero rows at the top
        zero_rows = np.all(image == 0, axis=1)
        image = np.vstack((image[zero_rows], image[~zero_rows]))

        # Convert the grayscale image into binary (0 or 1)
        return np.where(image > 0, 1, 0).astype(np.uint8)
