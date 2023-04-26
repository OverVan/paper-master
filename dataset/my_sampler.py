import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, labels2inds, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.catlocs = labels2inds

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            episode = []
            classes = np.random.choice(len(self.catlocs.keys()), self.n_cls, replace=False)
            for c in classes:
                inds = np.random.choice(self.catlocs[c], self.n_per, replace=False)
                episode.append(torch.from_numpy(inds))
            episode = torch.stack(episode)
            yield episode.view(-1)