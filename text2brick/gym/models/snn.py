import torch
from torch import nn
from text2brick.gym.models.cnn import CNN

# https://medium.com/@rinkinag24/a-comprehensive-guide-to-siamese-neural-networks-3358658c0513

class SNN(nn.Module):

    def __init__(self, image_target):
        super().__init__()

        self.cnn = CNN()
        self.target = self.cnn.forward(image_target)
        self.image_environement = None


    def forward(self, image_environement):

        self.image_environement = self.cnn.forward(image_environement)
        similarity = self._cosine_similarity(self.target, self.image_environement)
        distance = self._distance_euclidean(self.target, self.image_environement)
        return similarity, distance


    def _cosine_similarity(self, image1, image2):
        """
        Computes the cosine similarity between two feature maps.
        Args:
            image1 (torch.Tensor): First feature map of shape [C, H, W].
            image2 (torch.Tensor): Second feature map of shape [C, H, W].
        Returns:
            float: Cosine similarity value between the two feature maps.
        """
        # Flatten the feature maps
        image1_flat = image1.flatten(start_dim=0)
        image2_flat = image2.flatten(start_dim=0)

        # Compute cosine similarity
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
        # Flatten and normalize feature maps
        image1_flat = image1.flatten(start_dim=0)
        image2_flat = image2.flatten(start_dim=0)

        image1_norm = image1_flat / image1_flat.norm()
        image2_norm = image2_flat / image2_flat.norm()

        # Compute Euclidean distance
        distance = torch.norm(image1_norm - image2_norm, p=2)
        return distance.item()