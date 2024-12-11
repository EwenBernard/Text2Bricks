from text2brick.utils.ImageUtils import IoU
import numpy as np
from abc import ABC


class AbstractRewardFunc(ABC):
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs) -> float:
        raise NotImplementedError


class IoUValidityRewardFunc(AbstractRewardFunc): 
    def __init__(self, IoU_weight: float = 1.0 , validity_weight: float = 1.0):
        self.IoU_weight = IoU_weight
        self.validity_weight = validity_weight
    
    def __call__(self, target_img: np.array, world_img: np.array, is_brick_valid: bool) -> float:
        iou = IoU(target_img, world_img)
        reward = self.IoU_weight * iou + self.validity_weight * is_brick_valid
        return reward