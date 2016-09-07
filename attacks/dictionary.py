# coding: utf-8
from __future__ import division

'''
Implementation of a dictionary attack.

Assumes no bias has been added yet.
'''
import numpy as np
from helpers.logging import tls, log


@log
def apply(features, labels,
        ## params
        percentage_samples_poisoned,
        percentage_features_poisoned=.99,
        start=None,
        duration=None,
        ):
    '''
    Returns the input data with *replaced* data that is crafted specifically to
    cause a poisoning dictionary attack, where all features are set to one.

    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and  D: the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (-1 and 1)
    - percentage_samples_poisoned: float between 0 and 1
        percentage of the dataset under the attacker's control
    - percentage_features_poisoned: float between 0 and 1
        percentage of the features under the attacker's control
        (ie. attacker knowledge of features)

    Outputs:
    - X: poisoned features
    - Y: poisoned labels


    Notes:
    - if we assume the attacker is the same entity, then their knowledge
      itself will not vary. Ie. they will know the same X features, instead
      of for each email modelling a change in the X features they know.
    '''

    ## notations
    spam_label = 1
    X, Y = features, labels
    N, D = X.shape ## number of N: samples, D: features
    tls.logger.debug('X: (%s, %s)\tY: %s' % (N, D, str(Y.shape)))

    ## attack parameters
    start = int(N * start) if start else 0
    duration = int(N * duration) if duration else N
    end = start + duration if start + duration < N else N
    attack_range = np.arange(start, end)

    ## randomly replace some samples with the poisoned ones
    ## so that total number of samples doesn't change
    num_poisoned = int(len(attack_range) * percentage_samples_poisoned)
    tls.logger.debug('Amount poisoned: %s' % num_poisoned)
    poisoned_indices = np.random.choice(attack_range, num_poisoned, replace=False)
    tls.logger.debug('Poisoned indices: %s' % poisoned_indices)

    ## model attacker knowledge of feature space
    d_poisoned = int(D * percentage_features_poisoned)
    known_features = np.random.choice(D, d_poisoned, replace=False)
    X[np.ix_(poisoned_indices, known_features)] = 1
    tls.logger.debug('- one of the poisoned emails: %s =? %d (# features)' % (X[poisoned_indices[0]].sum(), D))

    ## the contamination assumption
    Y[poisoned_indices] = spam_label

    tls.logger.debug('- one of the poisoned emails\' label: %s =? 1' % Y[poisoned_indices[0]])
    ## TODO: these logging debugs should probably be asserts

    return X, Y

