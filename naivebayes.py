#!/usr/bin/env python
# coding: utf-8

'''
Inspired by Luis Munoz's MATLAB code for the Naive Bayes classifier model.
'''
import sys
import numpy as np


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
    tolerance = 1e-10 ## tolerance factor (to avoid under/overflows)

    ## estimate prior probability of spam class
    prior_spam = np.sum(Y == spam_label) / N
    prior__ham = 1 - prior_spam

    indices__ham = np.nonzero(Y == _ham_label)[0]
    indices_spam = np.nonzero(Y == spam_label)[0]
    N__ham = len(indices__ham)
    N_spam = len(indices_spam)

    ## estimate likelihood parameters for each class
    l__ham = np.sum(X[indices__ham], axis=0) / N__ham  ## presence of features in  ham class
    l_spam = np.sum(X[indices_spam], axis=0) / N_spam  ## presence of features in spam class

    l__ham, p_spam = map(lambda p: p.reshape((D, 1)), [l__ham, l_spam])
    l__ham, p_spam = map(process_parameters, [l__ham, l_spam])

    return prior__ham, prior_spam, l__ham, l_spam


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
    X, prior__ham, prior_spam, l__ham, l_spam = features, *parameters
    N, D = X.shape

    ## TODO is there a way to vectorise this ?
    predicted = np.zeros((N, 1)) ## prediction of class for each sample
    for i in range(1): ## i is the sample index

        ## apply model
        log_posterior__ham = np.log(prior__ham) +                \
                             np.dot(X[i, :], np.log(l__ham)) +   \
                             np.dot((1-X[i, :]), np.log(1-l__ham))
        log_posterior_spam = np.log(prior_spam)   +              \
                             np.dot(X[i, :], np.log(l_spam)) +   \
                             np.dot((1-X[i, :]), np.log(1-l_spam))

        ## calculate output
        ## assign class which is most likely over the other for sample i
        ## this works because labels are 0 and 1 for ham and spam respectively
        predicted[i] = (log_posterior_spam > log_posterior__ham)

    return predicted


def main():
    '''
    '''
    return


if __name__ == '__main__':
    sys.exit(main())
