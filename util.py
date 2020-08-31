"""
Massachusetts Institute of Technology

Izzy Brand, 2020
"""

from itertools import combinations_with_replacement
import numpy as np
import torch


def class_combinations(c, n, m=np.inf):
    """ Generates an array of n-element combinations where each element is one of
    the c classes (an integer). If m is provided and m < n^c, then instead of all
    n^c combinations, m combinations are randomly sampled.

    Arguments:
        c {int} -- the number of classes
        n {int} -- the number of elements in each combination

    Keyword Arguments:
        m {int} -- the number of desired combinations (default: {np.inf})

    Returns:
        np.ndarry -- An [m x n] or [n^c x n] array of integers in [0, c)
    """

    if m < c**n:
        # randomly sample combinations
        return np.random.randint(c, size=(m, n))
    else:
        p_c = combinations_with_replacement(np.arange(c), n)
        return np.array(list(iter(p_c)), dtype=int)

def H(x):
    """ Compute the element-wise entropy of x

    Arguments:
        x {torch.Tensor} -- array of numbers

    Returns:
        torch.Tensor -- H(x)
    """
    return -x*torch.log(x)

def hasnan(x):
    return torch.isnan(x).any()
