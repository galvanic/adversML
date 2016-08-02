# coding: utf-8
from __future__ import division

'''
TODO ? give each experiment a UID
'''
import os
import pickle
import numpy as np
import pandas as pd
from pprint import pprint
from copy import deepcopy
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

from helpers.performance import get_error, get_FPR, get_FNR, get_ROC_AUC
from helpers.specs import generate_specs, prepare_spec


def perform_experiment(spec, infolder, verbose=True):
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
    if verbose: pprint(spec)

    spec = prepare_spec(spec)
    if verbose: pprint(spec)

    ifilepath = os.path.join(infolder, '%s' % spec['dataset_filename'])
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
    attack = spec['attack']['type']
    attack_params = spec['attack']['parameters']
    X_train, Y_train = attack.apply(features=X_train, labels=Y_train, **attack_params)

    ## prepare dataset
    add_bias = lambda x: np.insert(x, 0, values=1, axis=1) # add bias term
    if spec['add_bias']:
        X_train, X_test = map(add_bias, [X_train, X_test])

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

    if verbose: pprint(performance)
    if verbose: print()

    ## release memory
    del X
    del Y
    del X_train
    del Y_train
    del X_test
    del Y_test

    return performance


def perform_experiment_batch(parameter_ranges, fixed_parameters, infolder,
        use_threads=True, num_threads=8):
    '''

    Inputs:
    - infolder: path of directory where processed datasets are
    - parameter_ranges
    - fixed_parameters

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
    ## extract names to use later for DF
    dimension_names, keys, variations= zip(*((p['name'], tuple(p['key']), p['values']) for p in parameter_ranges))

    ## get all possible variations for specs
    specifications = generate_specs(zip(keys, variations), fixed_parameters)

    ## perform each experiment
    ## TODO incorporate infolder directly by changing the specs
    perform_exp = partial(perform_experiment, infolder=infolder, verbose=False)

    ## use threads
    if use_threads:
        with ThreadPool(processes=num_threads) as pool:
            results = pool.map(perform_exp, specifications)
    else:
        results = map(perform_exp, specifications)

    ## put into DataFrame for analysis
    idx = pd.MultiIndex.from_product(variations, names=dimension_names)
    df = pd.DataFrame.from_records(data=results, index=idx)
    df.columns.names = ['metrics']

    return df

