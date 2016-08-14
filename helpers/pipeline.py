# coding: utf-8
from __future__ import division
'''
'''
import os
import pickle
import numpy as np
from pprint import pformat

from helpers.logging import tls, log
from helpers.performance import get_error, get_FPR, get_FNR, get_ROC_AUC
from helpers.specs import prepare_spec


@log(get_experiment_id=lambda args: args[0]['experiment_id'])
def perform_experiment(spec, infolder):
    '''
    Returns the performance of the experiment defined by the given specification

    Inputs:
    - spec: specifications of the experiment
        A dictionary
    - infolder: folderpath of directory where input data is
    - index: pandas MultiIndex of the experiment batch context

    Outputs:
    - performance: metrics of the performance of the trained classifier
        under the specified attack
        A dictionary
    '''

    tls.logger.info('spec:\n%s\n' % pformat(spec))
    spec = prepare_spec(spec)
    tls.logger.info('prepared spec:\n%s\n' % pformat(spec))

    ifilepath = os.path.join(infolder, '%s' % spec['dataset_filename'])
    with open('%s-features.dat' % ifilepath, 'rb') as infile:
        X = pickle.load(infile)

    with open('%s-labels.dat' % ifilepath, 'rb') as infile:
        Y = pickle.load(infile)

    ## normalise Y labels to (-1, 1)
    if tuple(np.unique(Y)) == (0, 1):
        Y = np.array(Y, dtype=np.int8) * 2 - 1

    N, D = X.shape
    tls.logger.debug('X: (%s, %s)\tY: (%s, %s)' % (N, D, *Y.shape))

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
    add_bias = lambda x: np.insert(x, 0, values=1, axis=1) # add bias term
    if spec['add_bias']:
        X_train, X_test = map(add_bias, [X_train, X_test])
        tls.logger.info('- added bias')

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
    }
    tls.logger.info('performance:\n%s' % pformat(performance))

    ## release memory
    del X
    del Y
    del X_train
    del Y_train
    del X_test
    del Y_test

    return performance

