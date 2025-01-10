import torch
from torch import nn
import matplotlib.pyplot as plt

from text2brick.gym.models.CNNImg import CNN

# Reference: https://medium.com/@rinkinag24/a-comprehensive-guide-to-siamese-neural-networks-3358658c0513


class SNN(nn.Module):
    """
    Siamese Neural Network (SNN) for comparing images using a shared CNN feature extractor.
    It computes similarity metrics (cosine similarity and Euclidean distance) between
    features of a target image and an environment image.
    """

    def __init__(
            self,
            image_target: torch.Tensor = None,
            normalize: bool = False,
            amplification: bool = False,
            threshold_factor: float = 0.0,  # Multiplier for the threshold based on max value
            squeeze_output = False
            ) -> None:
        """
        Initializes the SNN with a target image.

        Args:
            image_target (torch.Tensor): Target image to compare with, of shape [C, H, W].
            normalize (bool): Whether to normalize the feature tensor. Default is True.
            amplification (bool): Whether to amplify (square) the feature differences. Default is True.
            threshold_factor (float): Factor to calculate the threshold based on the max value of the feature map. Default is 0.1.
            
        """
        super().__init__()

        self.normalize = normalize
        self.amplification = amplification
        self.threshold_factor = threshold_factor
        self.squeeze_output = squeeze_output

        self.ouput_size = (13, 13)

        self.cnn = CNN()
        self.target = None
        if image_target is not None:
            self.target = self.cnn.forward(image_target)


    def forward(self, image_environement: torch.Tensor, image_target: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass to compute similarity metrics between the target and environment images.
        
        Args:
            image_environement (torch.Tensor): Environment image to compare, of shape [C, H, W].
            image_target (torch.Tensor, optional): Target image to compare, of shape [C, H, W]. Default is None.
            normalize (bool): Whether to normalize the feature vectors before comparison. Default is True.
            amplification (bool): Whether to apply feature amplification (squaring the difference). Default is True.
            
        Returns:
            torch.Tensor: Difference (or similarity) between the target and environment image features.
        """
        image_environement = self.cnn.forward(image_environement)

        if self.target is None and image_target is not None:
            image_target = self.cnn.forward(image_target)
        elif self.target is not None:
            image_target = self.cnn.forward(self.target)
        
        difference = image_target - image_environement

        return self._post_process(difference)
    

    def _post_process(self, features: torch.Tensor) -> torch.Tensor:
        """
        Post-process the feature map by applying amplification, normalization, and thresholding.
        
        Args:
            features (torch.Tensor): The feature tensor to process.
        Returns:
            torch.Tensor: The processed feature tensor.
        """
        if self.amplification:
            features = features.pow(2)  # Amplify the features (square the difference)
        
        if self.normalize:
            norm = features.norm(p=2, dim=1, keepdim=True) + 1e-10
            features = features / norm  # Normalize the feature tensor
        
        # Calculate the threshold based on the maximum value in the feature map
        if self.threshold_factor != 0:
            max_value = features.abs().max()
            threshold = self.threshold_factor * max_value  # Set threshold as 0.1 * max_value
            features = torch.where(torch.abs(features) < threshold, torch.zeros_like(features), features)

        if self.squeeze_output:
            features = features.squeeze(0)
            
        return features


    def feature_map(self, features: torch.Tensor, title: str = "Feature map") -> None:
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

        # Plot the feature map
        fig, axe = plt.subplots(1, 1, figsize=(10, 10))
        cax = axe.imshow(features, cmap='viridis')
        axe.set_title(title)
        axe.axis('off')  # Turn off axes for better visualization

        # Add a colorbar
        cbar = fig.colorbar(cax, ax=axe, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Feature Intensity')

        plt.show()
