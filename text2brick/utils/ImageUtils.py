from PIL import Image
import numpy as np

def array_to_image(array):
    array = np.where(array > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(array, mode='L')


def image_upscale(image: Image, scale_factor: int):
    new_size = (image.width * scale_factor, image.height * scale_factor)
    return image.resize(new_size, resample=Image.NEAREST)


def IoU(image1, image2):
    intersection = np.logical_and(image1, image2).sum()
    union = np.logical_or(image1, image2).sum()

    if union == 0:
        return 0
    
    return intersection/union