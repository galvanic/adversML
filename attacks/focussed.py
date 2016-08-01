# coding: utf-8
from __future__ import division
'''
Implementation of the focussed attack, see description in the `apply` function.

TODO A lot of shared code with the ham attack (obviously since focussed attack
     is the ham attack where the only ham is the target email) - refactor
TODO see what I said in `select_using_MI`, take into account absent features too
'''
import numpy as np
from sklearn.metrics import mutual_info_score
import random


def select_using_MI(features, labels, threshold=0.01, ham_label=-1):
    '''
    Returns indices of the most salient features for the ham class, using a
    mutual information score between feature values and class label, and
    from the highest scoring, filtering the ones that are most present
    in spam relatively. This makes sense since we then use these indices
    to choose which features to turn on in emails

    TODO or I could keep all the highest MI score features, and if more
         present in ham, I set it to 1, else I set it to zero
         requires extra array somewhere though
         Thinking about it, this is already what is happening since the ham
         malicious instances already have all their features set to zero
         (but this proportion of features controlled by the attacker could
         vary ? depending on attacket's dataset knowledge ? ie. malicious
         instances' feature values could be initialised randomly, or drawn from a
         spammy dicstribution to mimick an email that still has a malicious
         potential (although this isn't necessary since we are doing a poison,
         not evasion attack); then compare this to initialised with 0s or with 1s
    '''
    X, Y = features, np.ravel(labels)
    N, D = X.shape
    d = int(D * threshold) ## percentile of features to keep
    ## TODO d should be percentage of target email NOT all features

    ## calculate frequency of feature presence relative to each class
    ham_freq  = np.mean(X[np.ravel(Y == ham_label)], axis=0)
    spam_freq = np.mean(X[np.ravel(Y != ham_label)], axis=0)

    ## calculate mutual information between features and labels
    MI_per_feature = (mutual_info_score(X[:, f], Y) for f in range(D))
    MI_per_feature = np.fromiter(MI_per_feature, dtype=np.float16)

    ## keep only salient features for ham (according to relative presence in that class)
    #MIs = MI_per_feature[ham_freq > spam_freq]
    MIs = MI_per_feature
    salient_indices = np.argpartition(MIs, -d)[-d:]
    ## ^ https://stackoverflow.com/questions/10337533/a-fast-way-to-find-the-largest-n-elements-in-an-numpy-array/20177786#20177786

    return salient_indices


def apply(features, labels,
        ## params
        percentage_samples_poisoned,
        percentage_features_poisoned=1.0,
        feature_selection_method=select_using_MI,
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

    ## select using Mutual Information method
    y = np.ones(Y.shape)
    y[target_index] = -1
    #salient_indices = select_using_MI(X, y, threshold=0.1)

    ## randomly replace some samples with the poisoned ones
    ## so that total number of samples doesn't change
    poisoned_indices = np.random.choice(N, num_poisoned, replace=False)
    X[poisoned_indices] = 0

    ## "turn on" features whose presence is indicative of target
    X[poisoned_indices][:, salient_indices] = 1

    ## the contamination assumption
    Y[poisoned_indices] = spam_label

    return X, Y

