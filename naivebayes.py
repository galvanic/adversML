#!/usr/bin/env python
# coding: utf-8
from __future__ import division

'''
Inspired by Luis Munoz's MATLAB code for the Naive Bayes classifier model.

/!\ run with python3
'''
import sys
import numpy as np
from performance import get_error, get_FPR, get_FNR


def process_parameters(p, tolerance=1e-10):
    '''
    Returns parameters where NaNs, zeros and ones have been modified to avoid
    under/overflows (??)
    Helper function for the training function.

    TODO write better docstring and explanation
    '''
    p[np.isnan(p)] = tolerance
    p[p == 0]      = tolerance
    p[p == 1]      = 1 - tolerance
    return p


def train_naivebayes(features, labels):
    '''
    Returns the parameters for a Naive Bayes model

    Logs are used because otherwise multiplications of very small numbers,
    which leads to problems of over/underflows

    TRAINING PHASE

    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and D:        the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (0 and 1)

    Outputs:
    - parameters
    '''
    ## setup
    X, Y = features, labels
    _ham_label = 0
    spam_label = 1
    N, D = X.shape    ## number of N: training samples, D: features
    tolerance = 1e-30 ## tolerance factor (to avoid under/overflows)

    ## estimate prior probability of spam class
    prior_spam = np.sum(Y == spam_label) / N
    prior__ham = 1 - prior_spam

    indices__ham = np.nonzero(Y == _ham_label)[0]
    indices_spam = np.nonzero(Y == spam_label)[0]
    N__ham = len(indices__ham)
    N_spam = len(indices_spam)

    ## estimate likelihood parameters for each class
    ## looks at presence of features in each class
    likeli__ham = np.sum(X[indices__ham], axis=0) / N__ham
    likeli_spam = np.sum(X[indices_spam], axis=0) / N_spam

    likeli__ham, likeli_spam = map(lambda p: p.reshape((D, 1)), [likeli__ham, likeli_spam])
    likeli__ham, likeli_spam = map(process_parameters, [likeli__ham, likeli_spam])

    return prior__ham, prior_spam, likeli__ham, likeli_spam


def test_naivebayes(parameters, features):
    '''
    TEST PHASE

    Inputs:
    - parameters
    - features

    Outputs:
    - predicted: labels
    '''
    ## notation
    X, prior__ham, prior_spam, likeli__ham, likeli_spam = features, *parameters
    N, D = X.shape

    ## apply model
    ## Bernouilli Naive Bayes, takes into account absence of a feature
    ## TODO figure out why log of posterior calculation is this
    log_posterior__ham = np.log(prior__ham) +                   \
                         np.dot(   X,  np.log(  likeli__ham)) + \
                         np.dot((1-X), np.log(1-likeli__ham))
    log_posterior_spam = np.log(prior_spam)   +                 \
                         np.dot(   X,  np.log(  likeli_spam)) + \
                         np.dot((1-X), np.log(1-likeli_spam))

    ## no need to normalise since we are just interested in which
    ## posterior is higher (ie. which label is most likely given the data)

    log_posterior__ham, log_posterior_spam = map(np.ravel,
        [log_posterior__ham, log_posterior_spam])
    ## calculate output
    ## assign class which is most likely over the other
    ## this works because labels are 0 and 1 for ham and spam respectively
    predicted = (log_posterior_spam > log_posterior__ham)

    return predicted


def main():
    '''
    test my implementation
    '''
    import pickle

    with open('../datasets/processed/trec2007-1607061515-features.dat', 'rb') as infile:
        X = pickle.load(infile)

    with open('../datasets/processed/trec2007-1607061515-labels.dat', 'rb') as infile:
        Y = pickle.load(infile)

    N, D = X.shape

    permutated_indices = np.random.permutation(N)
    X = X[permutated_indices]
    Y = Y[permutated_indices]

    N_train = int(np.round(N * 0.5))
    X_train = X[:N_train]
    Y_train = Y[:N_train]
    X_test = X[N_train:]
    Y_test = Y[N_train:]

    parameters = train_naivebayes(features=X_train, labels=Y_train)
    O_train = test_naivebayes(parameters=parameters, features=X_train)
    O_test = test_naivebayes(parameters=parameters, features=X_test)

    print('error training set:\t%.3f' % get_error(Y_train, O_train))
    print('error testing  set:\t%.3f' % get_error(Y_test, O_test))
    print('false positive rate:\t%.3f' % get_FPR(Y_test, O_test))
    print('false negative rate:\t%.3f' % get_FNR(Y_test, O_test))

    return



if __name__ == '__main__':
    sys.exit(main())
