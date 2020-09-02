"""
Massachusetts Institute of Technology

Izzy Brand, 2020
"""
import numpy as np
import torch
import torch.nn as nn

from util import *


class Aquirer:
    """ Base class for aquisition functions (for single elements)

    Initialized with a pool of data. Each time select is called, every element
    in the pool is scored (with the score function) and the highest scoring
    element is returned. Elements are returned without replacment
    """
    def __init__(self, pool_dataset, device):
        self.pool_dataset = pool_dataset
        # pull out the block of data in a tensor (no labels)
        self.pool_data = pool_dataset.dataset.data[pool_dataset.indices, None, ...].float().to(device)
        self.remaining_indices = torch.ones(len(pool_dataset), dtype=bool)

    @staticmethod
    def score(model, x):
        """ Parallezied aquisition scoring function

        Arguments:
            model {nn.Module} -- the NN
            x {torch.Tensor} -- datapoints to evaluate

        Returns:
            [torch.Tensor] -- a vector of aquisition scores
        """
        return torch.zeros(len(x))

    def select(self, model):
        # score every datapoint in the pool under the model
        scores = self.score(model, self.pool_data)
        # mask out the scores by remaining and choose the best one
        best_idx = torch.argmax(scores - np.inf*(~self.remaining_indices))
        # and mark that index as taken
        self.remaining_indices[best_idx] = False
        # and return the data at the chosen index
        return self.pool_dataset[best_idx]

    def select_batch(self, model, batch_size):
        data, target = zip(*[self.select(model) for _ in range(batch_size)])
        return torch.cat(data), torch.LongTensor(target)


class BALD(Aquirer):
    def __init__(self, pool_dataset, device):
        super(BALD, self).__init__(pool_dataset, device)

    @staticmethod
    def score(model, x, k=10):
        # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]

        with torch.no_grad():
            # take k monte-carlo samples of forward pass w/ dropout
            Y = torch.stack([model(x) for i in range(k)], dim=1)
            H1 = H(Y.mean(axis=1)).sum(axis=1)
            H2 = H(Y).sum(axis=(1,2))/k

            return H1 - H2


class Random(Aquirer):
    def __init__(self, pool_dataset, device):
        super(Random, self).__init__(pool_dataset, device)

    @staticmethod
    def score(model, _):
        return np.random.rand()


class BatchBALD(Aquirer):
    def __init__(self, pool_dataset, device):
        super(BatchBALD, self).__init__(pool_dataset, device)
        self.m = 1e5 # number of MC samples for labels

    def select_batch(self, model, batch_size, k=10):
        # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]

        c = 10 # number of classes

        # forward pass on the pool once to get class probabilities for each x
        with torch.no_grad():
            # produces a tensor of [N x k x c] where N is the pool size
            pool_p_y = torch.stack([model(self.pool_data) for i in range(k)], dim=1)

        # this only need to be calculated once so we pull it out of the loop
        H2 = H(pool_p_y).sum(axis=(1,2))/k

        # get all class combinations
        c_1_to_n = class_combinations(c, batch_size, self.m)

        # tensor of size [c^(n-1) x k]
        p_y_1_to_n_minus_1 = None

        batch_idxs = []
        for n in range(batch_size):
            # tensor of size [N x c x k]
            p_y_n = pool_p_y
            # tensor of size [N x c^n x k]
            p_y_1_to_n = torch.flatten(torch.einsum('ik,pjk->pijk', p_y_1_to_n_minus_1, p_y_n), 1, 2)\
                if p_y_1_to_n_minus_1 is not None else p_y_n

            # and compute the left entropy term
            H1 = H(p_y_1_to_n.mean(axis=2)).sum(axis=1)
            # scores is a vector of scores for each element in the pool.
            # mask by the remaining indices and find the highest scoring element
            scores = H1 - H2
            # print(scores)
            best_idx = torch.argmax(scores - np.inf*(~self.remaining_indices))
            # print(f'Best idx {best_idx}')
            batch_idxs.append(best_idx)
            # save the computation for the next batch
            p_y_1_to_n_minus_1 = p_y_1_to_n[best_idx]
            # remove the chosen element from the remaining indices mask
            self.remaining_indices[best_idx] = False

        data, target = zip(*[self.pool_dataset[i] for i in batch_idxs])
        # print(self.remaining_indices.sum())
        return torch.cat(data), torch.LongTensor(target)
