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
        percentage_features_poisoned=.99,
        start=None,
        duration=None,
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
    tls.logger.debug('X: (%s, %s)\tY: %s' % (N, D, str(Y.shape)))

    ## attack parameters
    if start:
        start = int(N * start)
        duration = int(N * duration) if duration else N
        end = start + duration if start + duration < N else N
        attack_range = np.arange(start, end+1)
    else:
        attack_range = N

    ## randomly replace some samples with the poisoned ones
    ## so that total number of samples doesn't change
    num_poisoned = int(len(attack_range) * percentage_samples_poisoned)
    tls.logger.debug('Amount poisoned: %s' % num_poisoned)
    poisoned_indices = np.random.choice(attack_range, num_poisoned, replace=False)
    tls.logger.debug('Poisoned indices: %s' % poisoned_indices)
    X[poisoned_indices] = 0

    if not target:

        ## choose ham email
        if not target_index:
            target_index = np.random.choice(np.ravel(np.where(Y == ham_label)))

        target = X[target_index]

    ## find the most salient features, indicative of the target
    ## select most present method
    ## same tokens as email (only feature presence is taken into account)
    ## in this case, it is the email
    salient_mask = (target == 1)
    salient_indices = np.ravel(np.where(salient_mask))
    tls.logger.info('Salient indices: %s' % salient_indices)

    ## model attacker knowledge of benign class' feature space
    d = len(salient_indices)
    d_poisoned = int(d * percentage_features_poisoned)
    try:
        known_features = np.random.choice(salient_indices, d_poisoned, replace=False)
    except ValueError:
        ## too little info about the targeted email is known, not worth
        ## continuing experiment, TODO raise custom Exception
        known_features = []

    ## "turn on" features whose presence is indicative of ham
    X[np.ix_(poisoned_indices, known_features)] = 1

    ## the contamination assumption
    Y[poisoned_indices] = spam_label
    tls.logger.debug('- one of the poisoned emails\' label: %s =? 1' % Y[poisoned_indices[0]])

    return X, Y

