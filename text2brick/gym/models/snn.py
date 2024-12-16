import torch
from torch import nn
from text2brick.gym.models.cnn import CNN

# Reference: https://medium.com/@rinkinag24/a-comprehensive-guide-to-siamese-neural-networks-3358658c0513

class SNN(nn.Module):
    """
    Siamese Neural Network (SNN) for comparing images using a shared CNN feature extractor.
    It computes similarity metrics (cosine similarity and Euclidean distance) between
    features of a target image and an environment image.
    """

    def __init__(self, image_target):
        """
        Initializes the SNN with a target image.

        Args:
            image_target (torch.Tensor): Target image to compare with, of shape [C, H, W].
        """
        super().__init__()

        self.cnn = CNN()
        self.target = self.cnn.forward(image_target)
        self.image_environement = None


    def forward(self, image_environement):
        """
        Forward pass to compute similarity metrics between the target and environment images.

        Args:
            image_environement (torch.Tensor): Environment image to compare, of shape [C, H, W].

        Returns:
            tuple: (cosine similarity, normalized Euclidean distance) between the target and environment image features.
        """
        self.image_environement = self.cnn.forward(image_environement)

        return self.target - self.image_environement


    def get_features(self):
        """
        Get the extracted features for both the target and environment images.

        Returns:
            tuple: (target features, environment features).
        """
        return self.target, self.image_environement


    def _cosine_similarity(self, image1, image2):
        """
        Computes the cosine similarity between two feature maps.

        Args:
            image1 (torch.Tensor): First feature map of shape [C, H, W].
            image2 (torch.Tensor): Second feature map of shape [C, H, W].

        Returns:
            float: Cosine similarity value between the two feature maps.
        """
        # Flatten the feature maps into 1D tensors
        image1_flat = image1.flatten(start_dim=0)
        image2_flat = image2.flatten(start_dim=0)

        similarity = torch.nn.functional.cosine_similarity(image1_flat.unsqueeze(0), image2_flat.unsqueeze(0))
        return similarity.item()


    def _distance_euclidean(self, image1, image2):
        """
        Computes the normalized Euclidean distance between two feature maps.

        Args:
            image1 (torch.Tensor): First feature map of shape [C, H, W].
            image2 (torch.Tensor): Second feature map of shape [C, H, W].

        Returns:
            float: Normalized Euclidean distance.
        """
        # Flatten the feature maps into 1D tensors
        image1_flat = image1.flatten(start_dim=0)
        image2_flat = image2.flatten(start_dim=0)

        # Normalize each feature map to unit length
        image1_norm = image1_flat / image1_flat.norm()
        image2_norm = image2_flat / image2_flat.norm()

        distance = torch.norm(image1_norm - image2_norm, p=2)
        return distance.item()