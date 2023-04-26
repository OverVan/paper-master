import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .model import register, make
from utils import cosine_similarity


@register("adapt_proto")
class ProtoInduction(nn.Module):
    """
    想法一：原型（支持集）靠近查询集
    """

    def __init__(self, encoder, tao=10.):
        super(ProtoInduction, self).__init__()
        self.encoder = make(encoder["name"], **encoder["args"])
        self.tao = nn.Parameter(torch.tensor(tao))
        
    def forward(self, x, n, k, q):
        x = self.encoder(x)[0] if isinstance(self.encoder(x), tuple) else self.encoder(x)
        x = x.reshape(n, k + q, -1)
        support, query = torch.split(x, [k, q], dim=1)
        query = query.reshape(n * q, -1)
        proto = support.mean(1)
        adapted_proto = torch.zeros_like(proto)
        pre_sim = cosine_similarity(query, proto)
        pre_label = torch.argmax(pre_sim, 1)
        whole_label = [np.argwhere(pre_label.cpu() == cls).reshape(-1) for cls in range(n)]
        for cls in range(n):
            ref_ind = whole_label[cls]
            ref_query = query[ref_ind]
            ref_sim = pre_sim[ref_ind, cls]
            self_proto = proto[cls].unsqueeze(0)
            self_sim = cosine_similarity(proto[cls].unsqueeze(0), self_proto).squeeze(0)
            weight = F.softmax(torch.cat([ref_sim, self_sim]), dim=0)
            adapted_proto[cls] = torch.mul(weight.unsqueeze(1), torch.cat([ref_query, self_proto])).sum(0)
        sim = cosine_similarity(query, adapted_proto)
        return self.tao * sim
        