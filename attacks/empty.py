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
        percentage_features_poisoned=1.0,
        start=None,
        duration=None,
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
    IDEA after thinking about it, it doesn't make sense to vary attacker
         knowledge for the empty attack. For example, in the case of spam,
         they would just purely send an empty email. There would therefore
         not be *any* features turned on, by definition.
         You could imagine that there are features other than the email
         content, ie. the words (or wtv instance is being classified) that
         are taken into account (eg. meta-content, context-info, etc.) but
         that would be both very out of the attacker's control and especially,
         very specific, so there's no point modelling it as something random
         here, because these extra features I can think of are not random
         processes.

    TODO what happens if duration is longer than end of array ? WAAH
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
    X[poisoned_indices] = 0

    ## the contamination assumption
    Y[poisoned_indices] = spam_label

    tls.logger.debug('- one of the poisoned emails\' label: %s =? 1' % Y[poisoned_indices[0]])
    ## TODO: these logging debugs should probably be asserts

    return X, Y

