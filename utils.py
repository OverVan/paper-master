import os
import glob
import torch
import random
import scipy.stats
import numpy as np
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def log(content, path="train"):
    print(content)
    if not os.path.exists("./log"):
        os.mkdir("log")
    with open(os.path.join("log", path + ".txt"), "a", encoding="UTF-8") as file:
        print(content, file=file)
        

def euclidean_distance(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1), "feature dim should be equal"
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)
        
        
def cosine_similarity(x1, x2):
    assert x1.dim() == 2 and x2.dim() == 2, "2-d needed"
    assert x1.shape[1] == x2.shape[1], "feature dim should be equal"
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    return torch.mm(x1, x2.T)


def chebyshev_distance(x, y):
    assert x.dim() == 2 and y.dim() == 2, "2-d needed"
    assert x.shape[1] == y.shape[1], "feature dim should be equal"
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.max(torch.abs(x - y), dim=2)[0]


def manhattan_distance(x, y):
    assert x.dim() == 2 and y.dim() == 2, "2-d needed"
    assert x.shape[1] == y.shape[1], "feature dim should be equal"
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.sum(torch.abs(x - y), dim=2)


def make_ep_label(n, q):
    label = torch.arange(n).unsqueeze(1).expand(n, q).reshape(-1)
    return label.cuda()


def make_optimizer(name, params, lr, weight_decay):
    optimizers = {
        "SGD": SGD
    }
    return optimizers[name](params, lr, momentum=0.9, weight_decay=weight_decay)


def make_lr_scheduler(name, optimizer, milestones):
    schedulers = {
        "MultiStepLR": MultiStepLR
    }
    return schedulers[name](optimizer, milestones=milestones)


def fix_seed(seed=0):
    random.seed(seed)
    # 禁止hash随机化，使得实验可复现
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 针对多GPU
    torch.cuda.manual_seed_all(seed)


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None
    filelist = [x for x in filelist if os.path.basename(x) != 'best.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file