from PIL import Image
import numpy as np

def array_to_image(array: np.array) -> Image.Image:
    array = np.where(array > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(array, mode='L')


def image_upscale(image: Image.Image, scale_factor: int) -> Image.Image:
    new_size = (image.width * scale_factor, image.height * scale_factor)
    return image.resize(new_size, resample=Image.NEAREST)


def IoU(image1: np.array, image2: np.array) -> int:
    intersection = np.logical_and(image1, image2).sum()
    union = np.logical_or(image1, image2).sum()

    if union == 0:
        return 0
    
    return intersection/union