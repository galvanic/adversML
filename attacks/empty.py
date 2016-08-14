# coding: utf-8
from __future__ import division
'''
Implementation of an empty attack.

Assumes no bias has been added yet.
'''
import numpy as np
from helpers.logging import tls, log


@log
def apply(features, labels,
        ## params
        percentage_samples_poisoned,
        ):
    '''
    Returns the input data with *replaced* data that is crafted specifically to
    cause a poisoning empty attack, where all features are set to zero.

    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and  D: the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (-1 and 1)
    - percentage_samples_poisoned: float between 0 and 1
        percentage of the dataset under the attacker's control

    Outputs:
    - X: poisoned features
    - Y: poisoned labels

    TODO vary attacker knowledge (=influence over features)
    '''

    ## notations
    spam_label = 1
    X, Y = features, labels
    N, D = X.shape ## number of N: samples, D: features
    num_poisoned = int(N * percentage_samples_poisoned)

    tls.logger.debug('X: (%s, %s)\tY: (%s, %s)' % (N, D, *Y.shape))
    tls.logger.debug('Amount poisoned: %s' % num_poisoned)

    ## randomly replace some samples with the poisoned ones
    ## so that total number of samples doesn't change
    poisoned_indices = np.random.choice(N, num_poisoned, replace=False)
    X[poisoned_indices] = 0

    tls.logger.debug('Poisoned indices: %s' % poisoned_indices)

    ## the contamination assumption
    Y[poisoned_indices] = spam_label

    tls.logger.debug('- one of the poisoned emails\' label: %s =? 1' % Y[poisoned_indices[0]])
    ## TODO: these logging debugs should probably be asserts

    return X, Y

