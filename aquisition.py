"""
Massachusetts Institute of Technology

Izzy Brand, 2020
"""

import numpy as np
import torch
import torch.nn as nn

def H(x):
    return -x*torch.log(x)


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
        scores = self.score(model, self.pool_data)
        # mask out the scores by remaining and choose the best one
        best_idx = torch.argmax(scores * self.remaining_indices)
        best_x = self.pool_dataset[best_idx]
        self.remaining_indices[best_idx] = 0
        return best_x

class BALD(Aquirer):
    def __init__(self, pool_dataset, device):
        super(BALD, self).__init__(pool_dataset, device)

    @staticmethod
    def score(model, x, k=100):
        # I(y;W | x) = H(y|x) - E_w[ H(y|x,W) ]

        model.train() # make sure dropout is enabled
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

# class BatchAquirer:
#     def __init__(self, pool_dataset):
#         self.pool_dataset = pool_dataset
#         self.remaining_indices = torch.ones(len(pool_dataset), dtype=bool)

#     @staticmethod
#     def score(model, batch, x):
#         return torch.zeros(len(x))

#     def select(self, model, batch_size):
#         batch = []
#         batch_idxs = []
#         for _ in range(batch_size):
#             scores = [self.score(model, batch, x) for x,y in self.pool_dataset]
#             best_idx = np.argmax(np.array(scores)*self.remaining_indices)
#             batch_idxs.append(best_idx)
#             batch.append(self.pool_dataset[best_idx])

#         self.remaining_indices[best_idxs] = 0
#         return batch



# class BatchRandom(BatchAquirer):
#     def __init__(self, pool_dataset):
#         super(BatchRandom, self).__init__(pool_dataset)

#     @staticmethod
#     def score(model, ):
#         return np.random.rand()


# def BatchBALD_score(model, x, k=100):
#     model.train() # make sure dropout is enabled
#     with torch.no_grad():
#         # take k monte-carlo samples of forward pass w/ dropout
#         Y = torch.cat([model(x) for i in range(k)])

#         H1 = H(Y.mean(axis=0)).sum()
#         H2 = H(Y).sum()/k
