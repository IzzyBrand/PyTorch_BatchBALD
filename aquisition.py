""" 
Massachusetts Institute of Technology

Izzy Brand, 2020
"""

import numpy as np
import torch

def H(x):
    return -x*torch.log(x)


class Aquirer:
    """ Base class for aquisition functions (for single elements)

    Initialized with a pool of data. Each time select is called, every element
    in the pool is scored (with the score function) and the highest scoring
    element is returned. Elements are returned without replacment
    """
    def __init__(self, pool_data):
        self.pool_data = pool_data
        self.remaining_indices = np.ones(len(pool_data), dtype=bool)

    @staticmethod
    def score(model, x):
        return 0

    def select(self, model):
        scores = [self.score(model, x) for x,y in self.pool_data]
        best_idx = np.argmax(np.array(scores)*self.remaining_indices)
        best_x = self.pool_data[best_idx]
        self.remaining_indices[best_idx] = 0
        return best_x

class BatchAquirer:
    def __init__(self, pool_data):
        self.pool_data = pool_data
        self.remaining_indices = np.ones(len(pool_data), dtype=bool)

    @staticmethod
    def score(model, batch, x):
        return 0

    def select(self, model, batch_size):
        batch = []
        batch_idxs = []
        for _ in range(batch_size):
            scores = [self.score(model, batch, x) for x,y in self.pool_data]
            best_idx = np.argmax(np.array(scores)*self.remaining_indices)
            batch_idxs.append(best_idx)
            batch.append(self.pool_data[best_idx])

        self.remaining_indices[best_idxs] = 0
        return batch

class BALD(Aquirer):
    def __init__(self, pool_data):
        super(BALD, self).__init__(pool_data)

    @staticmethod
    def score(model, x, k=100):
        # I(y;W | x) = H(y|x) - E_w[ H(y|x,W) ]

        model.train() # make sure dropout is enabled
        with torch.no_grad():
            # take k monte-carlo samples of forward pass w/ dropout
            Y = torch.cat([model(x) for i in range(k)])

            H1 = H(Y.mean(axis=0)).sum()
            H2 = H(Y).sum()/k

            return H1 - H2


class Random(Aquirer):
    def __init__(self, pool_data):
        super(Random, self).__init__(pool_data)

    @staticmethod
    def score(model, _):
        return np.random.rand()



# def BatchBALD_score(model, x, k=100):
#     model.train() # make sure dropout is enabled
#     with torch.no_grad():
#         # take k monte-carlo samples of forward pass w/ dropout
#         Y = torch.cat([model(x) for i in range(k)])

#         H1 = H(Y.mean(axis=0)).sum()
#         H2 = H(Y).sum()/k
