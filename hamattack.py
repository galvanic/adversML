# coding: utf-8
from __future__ import division

'''
'''
import numpy as np

def apply(features, labels,
        ## params
        percentage_samples_poisoned,
        percentage_features_poisoned,
        feature_selection_method=None,
        ):
    '''
    Returns the input data with *added* data that is crafted specifically to
    cause a poisoning ham attack, where features of the contaminating emails
    are selected because they are indicative of the ham class.

    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and  D: the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (-1 and 1)
    - percentage_samples_poisoned: float between 0 and 1
        percentage of the dataset under the attacker's control
    - percentage_features_poisoned: float between 0 and 1
        percentage of the features under the attacker's control
    - feature_selection_method: string

    Outputs:
    - X: poisoned features
    - Y: poisoned labels

    TODO not sure if poisoned data is added to dataset or replaces samples in
         the dataset
    '''
    ## notations
    x, y = features, labels
    N, D = features.shape ## number of N: samples, D: features
    num_poisoned = int(N * percentage_samples_poisoned)

    ## find the most salient (positive) features, indicative of the ham class
    ham_indices = y[y == 0]
    hams = x[ham_indices, :][0, :, :]

    if not feature_selection_method:
        count = np.sum(hams, axis=0)
        salient = (count != 0) ## feature indices that are salient for ham class

    ham_attack = np.zeros((1, D))
    ham_attack[:, salient] = 1
    ham_attack = np.array([ham_attack,] * num_poisoned)
    ham_attack = ham_attack[:, 0, :]

    ## injection of poisoned features into training dataset
    X = np.append(features, ham_attack, axis=0)

    ## the contamination assumption
    poisoned_labels = np.ones((num_poisoned, 1))
    Y = np.append(labels, poisoned_labels, axis=0)

    return X, Y


