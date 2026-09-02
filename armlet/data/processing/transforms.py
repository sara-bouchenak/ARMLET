import torch
import torch.nn as nn


# Divide all elements of a tensor by the same factor
class DivTransform(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / self.factor


# Reshape a tensor, keeping the number of elements
class ReshapeTransform(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = list(size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(self.size).float()

class ToRGB(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, image):
        return image.convert("RGB")
