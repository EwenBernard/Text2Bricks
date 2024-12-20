import torch
from torch import nn
import matplotlib.pyplot as plt

#https://pytorch.org/vision/main/models/generated/torchvision.models.squeezenet1_0.html


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', weights="SqueezeNet1_0_Weights.IMAGENET1K_V1")
        self.model.eval()
        self.model = nn.Sequential(*list(self.model.features.children())[:13])
        self.layers = self.model.children()
        # for i, layer in enumerate(self.model.children()):
        #     print(f"Layer {i}: {layer}")
        self.parameters = sum(p.numel() for p in self.model.parameters())


    def forward(self, input_tensor):
        """
        Performs the forward pass by preprocessing the input image, running it through the model, 
        and returning the reduced 13x13 feature map.

        Args:
            image (Tensor): The input image to be processed by the model.
            
        Returns:
            torch.Tensor: The output tensor containing the reduced 13x13 feature map.
        """
        #TODO change cuda params 
        # Move the input and model to the GPU if available
        if torch.cuda.is_available():
            input_tensor = input_tensor.to('cuda')
            self.model.to('cuda')
        
        # Perform inference without computing gradients
        with torch.no_grad():
            output = self.model(input_tensor)

        # Combine the channels into a 13x13 tensor
        output_13x13 = output.mean(dim=1)

        #return output_13x13.squeeze(0)
        return output_13x13

    
    def feature_map(self, features):
        """
        Visualizes the combined feature map.

        Args:
            features (torch.Tensor): Feature map tensor of shape [H, W], where:
                                    H - height of the feature map,
                                    W - width of the feature map.
        """

        if features.dim() == 3:
            features = features.squeeze(0)

        if features.dim() != 2:
            raise ValueError(f"Expected a 2D tensor [H, W], but got shape {features.shape}")

        # Normalize the feature map to the range [0, 1]
        features_normalized = (features - features.min()) / (features.max() - features.min())

        # Plot the feature map
        fig, axe = plt.subplots(1, 1, figsize=(10, 10))
        axe.imshow(features_normalized, cmap='viridis')
        axe.set_title("Feature map")
        axe.axis('off')  # Turn off axes for better visualization
        plt.show()
