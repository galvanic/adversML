# coding: utf-8

'''
Implementation of an empty attack.

Assumes no bias has been added yet.
'''
import numpy as np

def apply_empty_attack(features, labels,
                       percentage_poisoned=0.1):
    '''
    Returns the input data with *added* data that is crafted specifically to
    cause a poisoning empty attack, where all features are set to zero.

    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and D:        the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (-1 and 1)
    - percentage_poisoned: float between 0 and 1
        percentage of the dataset under the attacker's control

    Outputs:
    - X: poisoned features
    - Y: poisoned labels

    TODO not sure if poisoned data is added to dataset or replaces samples in
         the dataset
    '''
    ## notations
    N, D = features.shape ## number of N: samples, D: features
    num_poisoned = int(N * percentage_poisoned)

    empty_features = np.zeros((num_poisoned, D))
    X = np.append(features, empty_features, axis=0)

    ## the contamination assumption
    poisoned_labels = np.ones((num_poisoned, 1))
    Y = np.append(labels, poisoned_labels, axis=0)

    return X, Y

