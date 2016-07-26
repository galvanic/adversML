# coding: utf-8
from __future__ import division

'''
TODO ? give each experiment a UID
'''
import pickle
import numpy as np
import pandas as pd
from itertools import product
from collections import OrderedDict
from pprint import pprint
from copy import deepcopy

from helpers.gradientdescent import max_iters
from helpers.performance import get_error, get_FPR, get_FNR, get_ROC_AUC
from helpers.specs import generate_specs

from classifiers import adaline as AdalineClassifier
from classifiers import naivebayes as NaivebayesClassifier
from attacks import empty as EmptyAttack
from attacks import ham as HamAttack

class no_attack():
    def apply(features, labels, **kwargs):
        return features, labels

Classifiers = {
    'adaline':  AdalineClassifier,
    'naivebayes': NaivebayesClassifier,
}

Attacks = {
    'empty': EmptyAttack,
    'ham': HamAttack,
    'none': no_attack,
}


def prepare_specs(spec):
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
    spec['training_parameters']['ham_label'] = ham_label
    spec['testing_parameters' ]['ham_label'] = ham_label

    normalise_key = lambda k: k.lower().replace(' ', '')

    spec['classifier'] = Classifiers[normalise_key(spec['classifier'])]
    spec['add_bias'] = True if spec['classifier'] != NaivebayesClassifier else False

    attack = Attacks[normalise_key(spec['attack'])]
    attack = 'none' if not attack else attack
    spec['attack'] = attack

    return spec


def perform_experiment(experiment, verbose=False):
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
        'AUC': get_ROC_AUC(Y_test, O_test, **experiment['label_type']),
    }

    return performance


def perform_experiment_batch(parameter_dimensions, fixed_parameters):
    '''
    - specifications: details for how to carry out experiments, what
        parameters to use etc.

    TODO what parts of the experiment specs are tied together ? and can
         therefore be simplified ?
    TODO make sure iteration is last ? or take it as seperate argument and put
         put it last ? OrderedDict should take in order of experiments, or the
         experiment_dimensions dictionary should already be an instance of
         Orditeration is last ? or take it as seperate argument and put
         put it last ? OrderedDict should take in order of experiments, or the
         experiment_dimensions dictionary should already be an instance of
         OrderedDict
    '''
    parameter_dimensions = OrderedDict(sorted(parameter_dimensions.items(), key=lambda t: len(t[1])))

    ## get all possible variations for specs
    specifications = generate_specs(parameter_dimensions, fixed_parameters)
    specs = map(prepare_specs, specifications)
    results = list(map(perform_experiment, specs))

    ## put into DataFrame for analysis
    dimensions, variations = zip(*parameter_dimensions.items())
    dimension_names = [name[-1] if type(name) == tuple else name for name in dimensions]
    idx = pd.MultiIndex.from_product(variations, names=dimension_names)
    df = pd.DataFrame.from_records(data=results, index=idx)
    df.columns.names = ['metrics']

    return df

