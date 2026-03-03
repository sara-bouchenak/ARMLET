import torch


class BinaryHingeLoss(torch.nn.MarginRankingLoss):
    def __init__(self, reduction = 'mean'):
        super().__init__(margin = 1.0, reduction = reduction)

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        normalized_x = torch.sigmoid(x)
        zeros = torch.zeros(x.shape)
        normalized_y = 2*y - 1
        return super().__call__(normalized_x, zeros, normalized_y)
