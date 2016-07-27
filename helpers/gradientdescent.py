# coding: utf-8
from __future__ import division

import numpy as np


### Helper functions for termination conditions

def Counter(max_iterations):
    i = max_iterations
    while True:
        yield i
        i -= 1

def max_iters(max_iterations):
    '''Assumes max_iterations is a Natural Integer.'''
    counter = Counter(max_iterations)
    return lambda: not next(counter)


def get_cost(Y, O):
    '''
    Calculate cost using Means Squared

    TODO ask does this make sense in here or grouped with gradient descent ?
    '''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    cost = np.mean(np.square(Y - O))
    return cost

