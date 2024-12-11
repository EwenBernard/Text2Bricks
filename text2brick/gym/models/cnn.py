import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from text2brick.utils.ImageUtils import array_to_image

#https://pytorch.org/vision/main/models/generated/torchvision.models.squeezenet1_0.html


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),  # Resize the shorter side of the image to 256 pixels
            transforms.CenterCrop(224),  # Crop the center to 224x224, the required input size for SqueezeNet
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale images to 3 channels (RGB)
            transforms.ToTensor(),  # Convert the PIL image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensor using ImageNet stats
        ])

        # Load model and extract the feature extraction layers
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', weights="SqueezeNet1_0_Weights.IMAGENET1K_V1")
        self.model.eval()
        self.model = nn.Sequential(*list(self.model.features.children())[:13])
        self.layers = self.model.children


    def forward(self, image):
        """
        Performs the forward pass by preprocessing the input image, running it through the model, 
        and returning the feature map.

        Args:
            image (PIL.Image or Tensor): The input image to be processed by the model.
            
        Returns:
            torch.Tensor: The output tensor containing the feature map produced by the model.
        """
            
        # Preprocess the input image
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        # Move the input and model to the GPU if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')
        
        # Perform inference without computing gradients
        with torch.no_grad():
            output = self.model(input_batch)

        return output