import torch
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt

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
        self.parameters = sum(p.numel() for p in self.model.parameters())


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
            print("yes")
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')
        
        # Perform inference without computing gradients
        with torch.no_grad():
            output = self.model(input_batch)

        return output.squeeze(0)


    def feature_map_channel(self, features, channel_index):
        """
        Visualizes a single channel from the feature map.

        Args:
            features (torch.Tensor): Feature map tensor of shape [C, H, W].
            channel_index (int): Index of the channel to visualize.
        """
        features = features.permute(1, 2, 0)
        # Check if the channel index is valid
        if channel_index < 0 or channel_index >= features.shape[0]:
            raise ValueError(f"Invalid channel_index {channel_index}. Must be between 0 and {features.shape[0] - 1}.")

        # Normalize the selected channel for visualization
        features_normalized = (features - features.min()) / (features.max() - features.min())

        # Plot the channel
        fig, axe = plt.subplots(1, 1, figsize=(15, 5))
        axe.imshow(features_normalized[..., channel_index], cmap='viridis')
        axe.set_title(f"Channel {channel_index + 1}")
        plt.show()
        
    
    def feature_map(self, features):
        """
        Visualizes the combined feature map by averaging across all channels.

        Args:
            features (torch.Tensor): Feature map tensor of shape [H, W, C], where:
                                    H - height of the feature map,
                                    W - width of the feature map,
                                    C - number of channels.
        """
        features = features.permute(1, 2, 0)
        combined_features = features.mean(dim=2)
        features_normalized = (combined_features - combined_features.min()) / (combined_features.max() - combined_features.min())

        fig, axe = plt.subplots(1, 1, figsize=(15, 5))
        axe.imshow(features_normalized, cmap='viridis')
        axe.set_title(f"Feature map (all channels combined)")
        plt.show()
