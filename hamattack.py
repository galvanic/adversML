# coding: utf-8
from __future__ import division

'''
'''
import numpy as np

def apply(features, labels,
        ## params
        percentage_samples_poisoned,
        percentage_features_poisoned=1.0,
        feature_selection_method=None,
        ham_label=-1,
        ):
    '''
    Returns the input data with *replaced* data that is crafted specifically to
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
    '''
    ## notations
    X, Y = features, labels
    N, D = X.shape ## number of N: samples, D: features
    num_poisoned = int(N * percentage_samples_poisoned)

    ## find the most salient features, indicative of the ham class
    ham_mask = np.ravel(Y == ham_label)
    hams = X[ham_mask]

    if not feature_selection_method:
        count = np.sum(hams, axis=0)
        salient_mask = (count != 0) ## feature indices that are salient for ham class

    ## randomly replace some samples with the poisoned ones
    ## so that total number of samples doesn't change
    poisoned_indices = np.random.choice(N, num_poisoned)
    X[poisoned_indices] = 0
    X[poisoned_indices][:, salient_mask] = 1

    ## the contamination assumption
    Y[poisoned_indices] = 1


    return X, Y


