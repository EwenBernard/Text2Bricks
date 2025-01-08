from torchvision import transforms
import numpy as np
from PIL import Image
import torch

class PreprocessImage: 
    def __init__(self) -> None:
        self.preprocess = transforms.Compose([
            transforms.Resize(256),  # Resize the shorter side of the image to 256 pixels
            transforms.CenterCrop(224),  # Crop the center to 224x224, the required input size for SqueezeNet
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale images to 3 channels (RGB)
            transforms.ToTensor(),  # Convert the PIL image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensor using ImageNet stats
        ])

    def __call__(self, image: np.array) -> torch.Tensor:

        pil_image = Image.fromarray(image.astype('uint8'))
        return self.preprocess(pil_image)