import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .model import register, make
from utils import cosine_similarity


@register("adapt_query")
class QueryCluster(nn.Module):
    """
    想法二：查询集内聚
    """

    def __init__(self, encoder, tao=10.):
        super(QueryCluster, self).__init__()
        self.encoder = make(encoder["name"], **encoder["args"])
        self.tao = nn.Parameter(torch.tensor(tao))
        self.k = 9
        
    def forward(self, x, n, k, q):
        x = self.encoder(x)[0] if isinstance(self.encoder(x), tuple) else self.encoder(x)
        x = x.reshape(n, k + q, -1)
        support, query = torch.split(x, [k, q], dim=1)
        query = query.reshape(n * q, -1)
        adapted_query = torch.zeros_like(query)
        proto = support.mean(1)
        query_sim = cosine_similarity(query, query)
        _, k_ind = torch.topk(query_sim, self.k)
        mask = torch.zeros_like(query_sim)
        mask = mask.scatter(1, k_ind, 1)
        mask = mask + mask.T
        whole_ind = [np.argwhere(mask[query_id].cpu() == 2).reshape(-1) for query_id in range(len(query))]
        for query_id in range(len(query)):
            # 不可能有0个，要么1个要么多个
            select_ind = whole_ind[query_id]
            select_query = query[select_ind]
            select_sim = query_sim[query_id, select_ind]
            weight = F.softmax(select_sim, dim=0)
            adapted_query[query_id] = torch.mul(weight.unsqueeze(1), select_query).sum(0)
        sim = cosine_similarity(adapted_query, proto)
        return self.tao * sim