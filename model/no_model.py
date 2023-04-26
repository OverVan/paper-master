import torch
from torch import nn

from .model import register, make
from utils import cosine_similarity


@register("no_model")
class NoModel(nn.Module):
    """
    不考虑support，只考虑原型自身和预分类到的查询样本
    """

    def __init__(self, encoder, tao=10.):
        super(NoModel, self).__init__()
        self.encoder = make(encoder["name"], **encoder["args"])
        self.tao = nn.Parameter(torch.tensor(tao))
        
    def forward(self, x, n, k, q):
        # 如果是wrn，剁掉分类器
        x = self.encoder(x)[0] if isinstance(self.encoder(x), tuple) else self.encoder(x)
        x = x.reshape(n, k + q, -1)
        support, query = torch.split(x, [k, q], dim=1)
        query = query.reshape(n * q, -1)
        proto = support.mean(1)
        sim = cosine_similarity(query, proto)
        return self.tao * sim