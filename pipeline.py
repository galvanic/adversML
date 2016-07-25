#!/usr/bin/env python3
# coding: utf-8
from __future__ import division

'''
'''
import sys
import pickle
import numpy as np
from pprint import pprint

from helpers.gradientdescent import max_iters
from helpers.performance import get_error, get_FPR, get_FNR, get_AUC
from helpers.specs import generate_specs


def perform_experiment(experiment):
    '''
    Returns the performance of the experiment.

    Inputs:
    - experiment: specifications in a dictionary

    Outputs:
    - performance: dictionary
    '''

    ifilepath = '/home/justine/Dropbox/imperial/computing/thesis/datasets/processed/%s' % experiment['dataset_filename']
    with open('%s-features.dat' % ifilepath, 'rb') as infile:
        X = pickle.load(infile)

    with open('%s-labels.dat' % ifilepath, 'rb') as infile:
        Y = pickle.load(infile)

    ## normalise Y labels to (-1, 1)
    if tuple(np.unique(Y)) == (0, 1):
        Y = np.array(Y, dtype=np.int8) * 2 - 1

    N, D = X.shape

    ## split dataset into training and testing sets
    permutated_indices = np.random.permutation(N)
    X = X[permutated_indices]
    Y = Y[permutated_indices]

    N_train = int(np.round(N * 0.5))
    X_train = X[:N_train]
    Y_train = Y[:N_train]
    X_test  = X[N_train:]
    Y_test  = Y[N_train:]

    ## apply attack
    attack = experiment['attack']
    attack_params = experiment['attack_parameters']
    X_train, Y_train = attack.apply(features=X_train, labels=Y_train, **attack_params)

    ## prepare dataset
    add_bias = lambda x: np.insert(x, 0, values=1, axis=1) # add bias term
    if experiment['add_bias']:
        X_train, X_test = map(add_bias, [X_train, X_test])

    ## apply model
    classifier = experiment['classifier']
    train_params = experiment['training_parameters']
    test_params  = experiment['testing_parameters' ]

    ## training phase
    model_parameters = classifier.train(features=X_train, labels=Y_train, **train_params)
    O_train = classifier.test(parameters=model_parameters, features=X_train, **test_params)

    ## testing phase
    O_test = classifier.test(parameters=model_parameters, features=X_test, **test_params)

    ## measure performance
    performance = {
        'error_train': get_error(Y_train, O_train),
        'error_test': get_error(Y_test,  O_test),
        'FPR': get_FPR(Y_test, O_test, **experiment['label_type']),
        'FNR': get_FNR(Y_test, O_test, **experiment['label_type']),
        'AUC': get_AUC(Y_test, O_test, **experiment['label_type']),
    }

    return performance


def main():
    '''
    Test the pipeline

    - specifications: details for how to carry out experiments, what
        parameters to use etc.

    TODO what parts of the experiment specs are tied together ? and can
         therefore be simplified ?
    TODO decide how to implement repetitions of experiments ?
    '''
    specifications = generate_specs()
    results = map(perform_experiment, specifications)

    #TODO zip specifications and results together

    return list(results)


if __name__ == '__main__':
    sys.exit(main())

