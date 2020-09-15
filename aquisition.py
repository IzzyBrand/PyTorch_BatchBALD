"""
Massachusetts Institute of Technology

Izzy Brand, 2020
"""
import numpy as np
import torch
import torch.nn as nn

from util import *


class Aquirer:
    """ Base class for aquisition function
    """
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.processing_batch_size = 128
        self.device = device

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

    def select_batch(self, model, pool_data):
        # score every datapoint in the pool under the model
        pool_loader = torch.utils.data.DataLoader(pool_data,
            batch_size=self.processing_batch_size, pin_memory=True, shuffle=False)
        scores = torch.zeros(len(pool_data)).to(self.device)
        for batch_idx, (data, _) in enumerate(pool_loader):
            end_idx = batch_idx + data.shape[0]
            scores[batch_idx:end_idx] = self.score(model, data.to(self.device))

        best_local_indices = torch.argsort(scores)[-self.batch_size:]
        best_global_indices = np.array(pool_data.indices)[best_local_indices.cpu().numpy()]
        return best_global_indices


class BALD(Aquirer):
    def __init__(self, pool_data, device):
        super(BALD, self).__init__(pool_data, device)

    @staticmethod
    def score(model, x, k=100):
        # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]

        with torch.no_grad():
            # take k monte-carlo samples of forward pass w/ dropout
            Y = torch.stack([model(x) for i in range(k)], dim=1)
            H1 = H(Y.mean(axis=1)).sum(axis=1)
            H2 = H(Y).sum(axis=(1,2))/k

            return H1 - H2


class Random(Aquirer):
    def __init__(self, pool_data, device):
        super(Random, self).__init__(pool_data, device)

    @staticmethod
    def score(model, _):
        return np.random.rand()


class BatchBALD(Aquirer):
    def __init__(self, pool_data, device):
        super(BatchBALD, self).__init__(pool_data, device)
        self.m = 1e4 # number of MC samples for labels

    # def score(self, model, x, k=100):
    #     # forward pass on the pool once to get class probabilities for each x
    #     with torch.no_grad():
    #         # produces a tensor of [N x c x k] where N is the pool size
    #         pool_p_y = torch.stack([model(x) for i in range(k)], dim=1).permute(0,2,1)
    #         # tensor of size [N x m x l]
    #         p_y_n = pool_p_y[:, self.c_1_to_n[:, self.n], :]
    #         # tensor of size [N x m x k]
    #         self.p_y_1_to_n = torch.einsum('mk,pmk->pmk', self.p_y_1_to_n_minus_1, p_y_n)\
    #             if self.p_y_1_to_n_minus_1 is not None else p_y_n
    #         # and compute the left entropy term
    #         H1 = H(self.p_y_1_to_n.mean(axis=2)).sum(axis=1)
    #         # compute the right term
    #         H2 = H(pool_p_y).sum(axis=(1,2))/k
    #         return H1 - H2

    # def select_batch(self, model, batch_size):
    #     # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]
    #     c = 10 # number of classes
    #     # get all class combinations
    #     self.c_1_to_n = class_combinations(c, batch_size, self.m)
    #     # tensor of size [m x k]
    #     self.p_y_1_to_n_minus_1 = None
    #     data = []
    #     target = []
    #     for self.n in range(batch_size):
    #         i = self.select(model, return_idx=True)
    #         # save the computation for the next batch
    #         self.p_y_1_to_n_minus_1 = self.p_y_1_to_n[i]
    #         # and add the chosen datapoint to the batch
    #         data.append(pool_data[i][0])
    #         target.append(pool_data[i][1])

    #     return torch.cat(data), torch.LongTensor(target)

    def select_batch(self, model, batch_size, k=100):
        # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]

        c = 10 # number of classes

        # # forward pass on the pool once to get class probabilities for each x
        # with torch.no_grad():
        #     # produces a tensor of [N x c x k] where N is the pool size
        #     pool_p_y = torch.stack([model(self.pool_data) for i in range(k)], dim=1).permute(0,2,1)

         # forward pass on the pool once to get class probabilities for each x
        with torch.no_grad():
            pool_loader = torch.utils.data.DataLoader(pool_data,
                batch_size=self.processing_batch_size, pin_memory=True, shuffle=False)
            pool_p_y = torch.zeros(len(pool_data), c, k)
            for batch_idx, (data, _) in enumerate(pool_loader):
                end_idx = batch_idx + data.shape[0]
                pool_p_y[batch_idx:end_idx] = torch.stack([model(data.to(self.device)) for i in range(k)], dim=1).permute(0,2,1)

        # this only need to be calculated once so we pull it out of the loop
        H2 = (H(pool_p_y).sum(axis=(1,2))/k).to(self.device)

        # get all class combinations
        c_1_to_n = class_combinations(c, batch_size, self.m)

        # tensor of size [m x k]
        p_y_1_to_n_minus_1 = None

        batch_idxs = []
        for n in range(batch_size):
            # tensor of size [N x m x l]
            p_y_n = pool_p_y[:, c_1_to_n[:, n], :].to(self.device)
            # tensor of size [N x m x k]
            p_y_1_to_n = torch.einsum('mk,pmk->pmk', p_y_1_to_n_minus_1, p_y_n)\
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

        data, target = zip(*[pool_data[i] for i in batch_idxs])
        # print(self.remaining_indices.sum())
        return torch.cat(data), torch.LongTensor(target)
