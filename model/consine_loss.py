import torch.nn.functional as F
from torch import nn


class CosineLoss(nn.Module):
    """ 余弦相似度损失 """

    def __init__(self, reduction="mean") -> None:
        super(CosineLoss, self).__init__()
        self.reduction = reduction

    def forward(self, a, b):
        loss = (1 - F.cosine_similarity(a, b)).mean(0) if self.reduction == "mean" else (1 - F.cosine_similarity(a, b)).sum(0)
        return loss
