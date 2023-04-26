import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        """
        :param label: 全体样本的标签
        :param n_batch: 每轮的batch数
        :param n_cls: n
        :param n_per: k+q
        """
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            episode = []
            classes = np.random.choice(len(self.catlocs), self.n_cls, replace=False)
            for c in classes:
                inds = np.random.choice(self.catlocs[c], self.n_per, replace=False)
                episode.append(torch.from_numpy(inds))
            episode = torch.stack(episode)
            yield episode.view(-1)