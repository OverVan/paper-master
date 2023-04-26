import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .model import register, make
from utils import cosine_similarity


@register("paper_model")
class PaperModel(nn.Module):
    """
    不考虑support，只考虑原型自身和预分类到的查询样本
    """

    def __init__(self, k, encoder, tao=10.):
        super(PaperModel, self).__init__()
        self.encoder = make(encoder["name"], **encoder["args"])
        self.tao = nn.Parameter(torch.tensor(tao))
        self.k = k
        
    def forward(self, x, n, k, q):
        # 如果是wrn，剁掉分类器
        x = self.encoder(x)[0] if isinstance(self.encoder(x), tuple) else self.encoder(x)
        x = x.reshape(n, k + q, -1)
        support, query = torch.split(x, [k, q], dim=1)
        query = query.reshape(n * q, -1)
        proto = support.mean(1)
        adapted_proto = torch.zeros_like(proto)
        adapted_query = torch.zeros_like(query)
        pre_sim = cosine_similarity(query, proto)
        pre_label = torch.argmax(pre_sim, 1)
        whole_label = [np.argwhere(pre_label.cpu() == cls).reshape(-1) for cls in range(n)]
        for cls in range(n):
            ref_ind = whole_label[cls]
            # 一定二维或空，即使ref_ind只有一个或没有元素
            ref_query = query[ref_ind]
            # 这个一定是一维的
            ref_sim = pre_sim[ref_ind, cls]
            # 二维
            self_proto = proto[cls].unsqueeze(0)
            # 一维
            self_sim = cosine_similarity(proto[cls].unsqueeze(0), self_proto).squeeze(0)
            weight = F.softmax(torch.cat([ref_sim, self_sim]), dim=0)
            adapted_proto[cls] = torch.mul(weight.unsqueeze(1), torch.cat([ref_query, self_proto])).sum(0)
        query_sim = cosine_similarity(query, query)
        _, k_ind = torch.topk(query_sim, self.k)
        mask = torch.zeros_like(query_sim)
        mask = mask.scatter(1, k_ind, 1)
        mask = mask + mask.T
        whole_ind = [np.argwhere(mask[query_id].cpu() == 2).reshape(-1) for query_id in range(len(query))]
        for query_id in range(len(query)):
            select_ind = whole_ind[query_id]
            # 二维
            select_query = query[select_ind]
            # 一维
            select_sim = query_sim[query_id, select_ind]
            weight = F.softmax(select_sim, dim=0)
            adapted_query[query_id] = torch.mul(weight.unsqueeze(1), select_query).sum(0)
        sim = cosine_similarity(adapted_query, adapted_proto)
        return self.tao * sim