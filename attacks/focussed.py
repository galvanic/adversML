# coding: utf-8
from __future__ import division
'''
Implementation of the focussed attack, see description in the `apply` function.

TODO A lot of shared code with the ham attack (obviously since focussed attack
     is the ham attack where the only ham is the target email) - refactor
TODO see what I said in `select_using_MI`, take into account absent features too
'''
import numpy as np
from helpers.logging import tls, log


@log
def apply(features, labels,
        ## params
        percentage_samples_poisoned,
        percentage_features_poisoned=1.0,
        feature_selection_method=None,
        threshold=0.01,
        target=None,
        target_index=None,
        ham_label=-1,
        ):
    '''
    Returns the input data with *replaced* data that is crafted specifically to
    cause a poisoning focussed attack, where features of the contaminating emails
    are selected because they are indicative of a specific email (ie. attack is
    *focussed* on that email)

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
    - threshold: percentile of features to keep

    Outputs:
    - X: N * D poisoned features
    - Y: N * 1 poisoned labels

    TODO see what I said in `select_using_MI`, take into account absent features too
    '''

    ## notations
    spam_label = 1
    X, Y = features, labels
    N, D = X.shape                        ## number of N: samples, D: features
    num_poisoned = int(N * percentage_samples_poisoned)

    tls.logger.debug('X: (%s, %s)\tY: (%s, %s)' % (N, D, *Y.shape))
    tls.logger.debug('Amount poisoned: %s' % num_poisoned)

    if not target:

        ## choose ham email
        if not target_index:
            target_index = np.random.choice(np.nonzero(Y == ham_label)[0])

        target = X[target_index]

    ## find the most salient features, indicative of the target
    ## select most present method
    ## same tokens as email (only feature presence is taken into account)
    ## in this case, it is the email
    salient_mask = (target == 1)
    salient_indices = np.nonzero(salient_mask)[0]

    tls.logger.info('Salient indices: %s' % salient_indices)

    ## randomly replace some samples with the poisoned ones
    ## so that total number of samples doesn't change
    poisoned_indices = np.random.choice(N, num_poisoned, replace=False)
    X[poisoned_indices] = 0

    tls.logger.debug('Poisoned indices: %s' % poisoned_indices)

    ## "turn on" features whose presence is indicative of target
    X[np.ix_(poisoned_indices, salient_indices)] = 1

    ## the contamination assumption
    Y[poisoned_indices] = spam_label

    tls.logger.debug('- one of the poisoned emails\' label: %s =? 1' % Y[poisoned_indices[0]])

    return X, Y

