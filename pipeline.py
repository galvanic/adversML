#!/usr/bin/env python3
# coding: utf-8
from __future__ import division

'''
'''

import sys
import numpy as np
import pickle

## helper functions
from gradientdescent import max_iters
from performance import get_error, get_FPR, get_FNR

## classifier models
import adaline
import naivebayes

## attacks
import empty
import hamattack


def process_experiment_declaration(experiment):
    '''
    Returns the experiment dictionary specification ready to carry out the
    experiment.
    For example, it duplicates certain keys so that the user doesn't have to
    enter them more than once (would increase chance of errors) and replaces
    None by actual objects (like a function that does nothing for the empty
    attack but would have been faff for user to write).
    '''
    ham_label = experiment['label_type']['ham_label']
    experiment['training_parameters']['ham_label'] = ham_label
    experiment['testing_parameters' ]['ham_label'] = ham_label

    if not experiment['attack']:
        def no_attack(x, **kwargs):
            return x
        experiment['attack'] = no_attack
    return experiment


## helper functions to prepare dataset
add_bias = lambda x: np.insert(x, 0, values=1, axis=1) # add bias term
convert_labels = lambda y: y*2 - 1


def perform_experiment(experiment):
    '''
    Returns the performance of the experiment.

    Inputs:
    - experiment: specifications in a dictionary

    Outputs:
    - performance: dictionary
    '''

    ifilepath = '../datasets/processed/%s' % experiment['dataset_filename']
    with open('%s-features.dat' % ifilepath, 'rb') as infile:
        X = pickle.load(infile)

    with open('%s-labels.dat' % ifilepath, 'rb') as infile:
        Y = pickle.load(infile)

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
    X_test,  Y_test  = attack.apply(features=X_test,  labels=Y_test,  **attack_params)

    ## prepare dataset
    if experiment['classifier'] != naivebayes:
        X_train, X_test = map(add_bias, [X_train, X_test])
    if experiment['label_type']['ham_label'] == -1:
        Y_train, Y_test = map(convert_labels, [Y_train, Y_test])

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
        'error_test':  get_error(Y_test,  O_test),
        'FPR':         get_FPR(Y_test, O_test, **experiment['label_type']),
        'FNR':         get_FNR(Y_test, O_test, **experiment['label_type']),
    }

    return performance


def main():
    '''
    Test the pipeline
    '''
    experiments = [
        {
            'dataset': 'trec2007',
            'dataset_filename': 'trec2007-1607061515',
            'feature_extraction_parameters': {
            },
            'label_type': {
                'ham_label': -1,
                'spam_label': 1,
            },
            'attack': hamattack,
            'attack_parameters': {
                'percentage_samples_poisoned': 0.1,
            },
            'classifier': adaline,
            'training_parameters': {
                'learning_rate': 0.06,
                'initial_weights': None,
                'termination_condition': max_iters(40),
                'verbose': False,
            },
            'testing_parameters': {
            },
        },

        {
            'dataset': 'trec2007',
            'dataset_filename': 'trec2007-1607061515',
            'feature_extraction_parameters': {
            },
            'label_type': {
                'ham_label': -1,
                'spam_label': 1,
            },
            'attack': hamattack,
            'attack_parameters': {
                'percentage_samples_poisoned': 0.1,
            },
            'classifier': naivebayes,
            'training_parameters': {
            },
            'testing_parameters': {
            },
        },
    ]

    experiments = map(process_experiment_declaration, experiments)
    results = map(perform_experiment, experiments)

    from pprint import pprint
    pprint(list(results))

    return


if __name__ == '__main__':
    sys.exit(main())
