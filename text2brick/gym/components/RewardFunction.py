from text2brick.utils.ImageUtils import IoU
import numpy as np
from abc import ABC
from typing import Tuple


class AbstractRewardFunc(ABC):
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs) -> float:
        raise NotImplementedError


class IoUValidityRewardFunc(AbstractRewardFunc): 
    def __init__(self, IoU_weight: float = 0.5 , validity_weight: float = 1.0) -> None:
        self.IoU_weight = IoU_weight
        self.validity_weight = validity_weight
        self.last_iou = 0

    def __str__(self):
        return f"Reward function : {self.IoU_weight}*(IoU - last_IoU) + {self.validity_weight}*validity"
    

    def _crop(self, img: np.array, center: Tuple[int, int], roi_size: int) -> np.array:
        """
        Crops an image around the specified center with the given ROI size.

        Parameters:
        img (np.array): The input image as a NumPy array.
        center (Tuple[int, int]): The (x, y) coordinates of the center for cropping.
        roi_size (int): The size of the square ROI to crop.

        Returns:
        np.array: The cropped image as a NumPy array.
        """
        x, y = center
        half_size = roi_size // 2
        
        # Calculate the boundaries of the cropping box
        x_start = max(x - half_size, 0)
        x_end = min(x + half_size, img.shape[1])
        y_start = max(y - half_size, 0)
        y_end = min(y + half_size, img.shape[0])
        
        # Crop and return the image
        return img[y_start:y_end, x_start:x_end]

    
    def __call__(
            self,
            target_img: np.array,
            world_img: np.array,
            validity: int,
            center: Tuple[int, int] = None,
            roi_size: int = 0,
        ) -> Tuple[float, float]:
        """
        Computes the reward based on the Intersection over Union (IoU) and validity.
        Optionally crops the images before computation if center and roi_size are provided.

        Parameters:
        target_img (np.array): The target image as a NumPy array.
        world_img (np.array): The world image as a NumPy array.
        validity (int): The validity indicator (1 for valid, 0 for invalid).
        center (Tuple[int, int], optional): The (x, y) coordinates of the center for cropping.
        roi_size (int, optional): The size of the square ROI to crop.

        Returns:
        Tuple[float, float]: The calculated reward and IoU growth.
        """
        if center is not None and roi_size > 0:
            target_img = self._crop(target_img, center, roi_size)
            world_img = self._crop(world_img, center, roi_size)

        iou_diff = IoU(target_img, world_img) - self.last_iou

        # Set validity to -1 if invalid, otherwise 1
        validity = -1 if not validity else 1

        reward = self.IoU_weight * iou_diff + self.validity_weight * validity
        return reward, iou_diff