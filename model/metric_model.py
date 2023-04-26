import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .model import register, make
from utils import euclidean_distance, manhattan_distance, chebyshev_distance


@register("metric_model")
class MetricModel(nn.Module):
    def __init__(self, k, encoder, metric, tao=10.):
        super(MetricModel, self).__init__()
        self.encoder = make(encoder["name"], **encoder["args"])
        self.tao = nn.Parameter(torch.tensor(tao))
        self.metric = metric
        if self.metric == "euclidean_distance":
            self.tao = nn.Parameter(torch.tensor(.1))
        self.k = k
        
    def dist(self, query, proto):
        if self.metric == "euclidean_distance":
            dist = euclidean_distance(query, proto)
        elif self.metric == "manhattan_distance":
            dist = manhattan_distance(query, proto)
        elif self.metric == "chebyshev_distance":
            dist = chebyshev_distance(query, proto)
        return dist        
        
    def forward(self, x, n, k, q):
        x = self.encoder(x)[0] if isinstance(self.encoder(x), tuple) else self.encoder(x)
        x = x.reshape(n, k + q, -1)
        support, query = torch.split(x, [k, q], dim=1)
        query = query.reshape(n * q, -1)
        proto = support.mean(1)
        adapted_proto = torch.zeros_like(proto)
        adapted_query = torch.zeros_like(query)
        pre_sim = self.dist(query, proto)
        pre_label = torch.argmax(pre_sim, 1)
        whole_label = [np.argwhere(pre_label.cpu() == cls).reshape(-1) for cls in range(n)]
        for cls in range(n):
            ref_ind = whole_label[cls]
            ref_query = query[ref_ind]
            ref_sim = pre_sim[ref_ind, cls]
            self_proto = proto[cls].unsqueeze(0)
            self_sim = self.dist(proto[cls].unsqueeze(0), self_proto).squeeze(0)
            weight = F.softmax(torch.cat([ref_sim, self_sim]), dim=0)
            adapted_proto[cls] = torch.mul(weight.unsqueeze(1), torch.cat([ref_query, self_proto])).sum(0)
        query_sim = self.dist(query, query)
        _, k_ind = torch.topk(query_sim, self.k)
        mask = torch.zeros_like(query_sim)
        mask = mask.scatter(1, k_ind, 1)
        mask = mask + mask.T
        whole_ind = [np.argwhere(mask[query_id].cpu() == 2).reshape(-1) for query_id in range(len(query))]
        for query_id in range(len(query)):
            select_ind = whole_ind[query_id]
            select_query = query[select_ind]
            select_sim = query_sim[query_id, select_ind]
            weight = F.softmax(select_sim, dim=0)
            adapted_query[query_id] = torch.mul(weight.unsqueeze(1), select_query).sum(0)
        return self.tao * self.dist(adapted_query, adapted_proto)