# coding: utf-8
from __future__ import division
'''
'''
import os
import pickle
import numpy as np
import pandas as pd
from pprint import pformat

from helpers.logging import tls, log
from helpers.performance import get_error, get_FPR, get_FNR, get_ROC_AUC, get_TP, get_TN, get_FP, get_FN


## helpers
add_bias = lambda x: np.insert(x, 0, values=1, axis=1) # add bias term


@log(get_experiment_id=lambda args: args[0]['experiment_id'])
def run_experiment(spec):
    '''
    Returns the performance of the experiment defined by the given specification

    Inputs:
    - spec: specifications of the experiment
        A dictionary

    Outputs:
    - performance: metrics of the performance of the trained classifier
        under the specified attack
        A dictionary
    '''

    tls.logger.info('spec:\n%s\n' % pformat(spec))
    spec = prepare_spec(spec)
    tls.logger.info('prepared spec:\n%s\n' % pformat(spec))

    ifilepath = os.path.join(spec['dataset_dirpath'], spec['dataset_filename'])
    with open('%s-features.dat' % ifilepath, 'rb') as infile:
        X = pickle.load(infile)

    with open('%s-labels.dat' % ifilepath, 'rb') as infile:
        Y = pickle.load(infile)

    ## normalise Y labels to (-1, 1)
    if tuple(np.unique(Y)) == (0, 1):
        Y = np.array(Y, dtype=np.int8) * 2 - 1

    N, D = X.shape
    tls.logger.debug('X: (%s, %s)\tY: %s' % (N, D, str(Y.shape)))

    ## split dataset into training and testing sets
    permuted_indices = np.random.permutation(N)
    X = X[permuted_indices]
    Y = Y[permuted_indices]

    N_train = int(np.round(N * 0.5))
    X_train = X[:N_train]
    Y_train = Y[:N_train]
    X_test  = X[N_train:]
    Y_test  = Y[N_train:]

    ## apply attack to training set (=poisoning attack)
    attack = spec['attack']['type']
    attack_params = spec['attack']['parameters']
    if attack_params['percentage_samples_poisoned'] != 0:
        X_train, Y_train = attack.apply(features=X_train, labels=Y_train, **attack_params)

    ## prepare dataset
    if spec['add_bias']:
        X_train, X_test = map(add_bias, [X_train, X_test])
        tls.logger.debug('- added bias')

    ## apply model
    classifier = spec['classifier']['type']
    train_params = spec['classifier']['training_parameters']
    test_params  = spec['classifier']['testing_parameters' ]

    ## training phase
    model_parameters = classifier.fit(features=X_train, labels=Y_train, **train_params)
    O_train = classifier.predict(parameters=model_parameters, features=X_train, **test_params)

    ## testing phase
    O_test = classifier.predict(parameters=model_parameters, features=X_test, **test_params)

    ## measure performance
    performance = {
        'error_train': get_error(Y_train, O_train),
        'error_test': get_error(Y_test,  O_test),
        'FPR': get_FPR(Y_test, O_test, **spec['label_type']),
        'FNR': get_FNR(Y_test, O_test, **spec['label_type']),
        'AUC': get_ROC_AUC(Y_test, O_test, **spec['label_type']),
        'TP': get_TP(Y_test, O_test, **spec['label_type']),
        'TN': get_TN(Y_test, O_test, **spec['label_type']),
        'FP': get_FP(Y_test, O_test, **spec['label_type']),
        'FN': get_FN(Y_test, O_test, **spec['label_type']),
    }
    tls.logger.info('performance:\n%s' % pformat(performance))
    df_row = pd.DataFrame.from_records([performance])
    df_row.columns = df_row.columns.set_names(['metrics'])

    ## release memory
    del X
    del Y
    del X_train
    del Y_train
    del X_test
    del Y_test

    return df_row


from copy import deepcopy

from classifiers import adaline as AdalineClassifier
from classifiers import naivebayes as NaivebayesClassifier
from classifiers import logistic_regression as LogisticRegressionClassifier

from attacks import empty as EmptyAttack
from attacks import ham as HamAttack
from attacks import dictionary as DictionaryAttack
from attacks import focussed as FocussedAttack

class no_attack():
    def apply(features, labels, **kwargs):
        return features, labels

Classifiers = {
    'adaline':  AdalineClassifier,
    'naivebayes': NaivebayesClassifier,
    'logisticregression': LogisticRegressionClassifier,
}

Attacks = {
    'empty': EmptyAttack,
    'ham': HamAttack,
    'dictionary': DictionaryAttack,
    'focussed': FocussedAttack,
    'none': no_attack,
}


def prepare_spec(spec):
    '''
    Returns the experiment dictionary specification ready to carry out the
    experiment.
    For example, it duplicates certain keys so that the user doesn't have to
    enter them more than once (would increase chance of errors) and replaces
    None by actual objects (like a function that does nothing for the empty
    attack but would have been faff for user to write).

    TODO raise exceptions if doesn't exist, or catch KeyError
    '''
    spec = deepcopy(spec)

    ham_label = spec['label_type']['ham_label']
    spec['classifier']['training_parameters']['ham_label'] = ham_label
    spec['classifier']['testing_parameters' ]['ham_label'] = ham_label

    normalise_key = lambda k: k.lower().replace(' ', '')

    ## classifier
    classifier = spec['classifier']['type']
    classifier = Classifiers[normalise_key(classifier)]
    spec['add_bias'] = True if classifier != NaivebayesClassifier else False
    spec['classifier']['type'] = classifier

    ## attack
    attack = spec['attack']['type']
    attack = 'none' if not attack else attack
    attack = Attacks[normalise_key(attack)]
    spec['attack']['type'] = attack

    return spec

