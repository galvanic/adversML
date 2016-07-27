# coding: utf-8
from __future__ import division

'''
Implementation of a dictionary attack.

Assumes no bias has been added yet.
'''
import numpy as np

def apply(features, labels,
        ## params
        percentage_samples_poisoned,
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

    ## randomly replace some samples with the poisoned ones
    ## so that total number of samples doesn't change
    poisoned_indices = np.random.choice(N, num_poisoned)
    X[poisoned_indices] = 1

    ## the contamination assumption
    Y[poisoned_indices] = spam_label

    return X, Y

