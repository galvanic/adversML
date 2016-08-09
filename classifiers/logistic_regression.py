# coding: utf-8
from __future__ import division
'''
Implementation of logistic regression
Adapted from: https://github.com/kaylashapiro/SpamFilter/blob/master/Code/logisticReg.py
'''
import logging
LOGGER = logging.getLogger(__name__)

import numpy as np

from helpers.gradientdescent import max_iters, get_cost
from helpers.performance import get_error

## helpers
sigmoid = lambda z: 1 / (1 + np.exp(-z))
tanh = lambda z: np.tanh(z)


def fit(features, labels,
        ## params:
        initial_weights=None,
        learning_rate=0.1,
        max_epochs=5,
        ham_label=-1,
        spam_label=1,
        ):
    '''
    Returns

    TRAINING PHASE

    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and  D: the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (-1 and 1)
    - initial_weights: D * 1 Numpy vector, beginning weights
    - learning_rate: learning rate, a float between 0 and 1
    - termination_condition: returns a bool

    Output:
    - W: D * 1 Numpy vector of real values
    '''
    LOGGER.info('Training Logistic Regression classifier')

    ## notation
    X, Y = features, labels
    N, D = X.shape           # N #training samples; D #features
    LOGGER.debug('X: (%s, %s)\tY: (%s, %s)' % (N, D, *Y.shape))

    ## initialise weights
    W = np.zeros((D, 1)) if initial_weights is None else initial_weights.reshape((D, 1))
    LOGGER.debug('initial weights: %s' % np.ravel(W))

    ## evaluate the termination condition
    for epoch in range(max_epochs):
        LOGGER.info('epoch %d:' % epoch)

        permutated_indices = np.random.permutation(N)
        X = X[permutated_indices]
        Y = Y[permutated_indices]

        ## stochastic gradient descent, sample by sample
        for sample in range(N):
            logging.debug('- sample %d:' % sample)
            x, y = X[sample], Y[sample]

            ## classifier output of current epoch
            o = tanh(np.dot(x, W))
            LOGGER.debug('-- output: %s' % o)

            ## gradient descent: calculate gradient
            gradient = np.multiply((o - y), x)
            LOGGER.debug('-- gradient: %s' % gradient)

            ## update weights
            W = W - learning_rate * gradient.reshape(W.shape)
            LOGGER.debug('-- weights: %s' % np.ravel(W))

        P = predict(W, X)
        error = get_error(Y, P)
        cost = get_cost(Y, P)
        LOGGER.info('- cost = %.2E' % cost)
        LOGGER.info('- error = %.2f' % error)

    return W


def predict(parameters, features,
        ## params
        ham_label=-1,
        spam_label=1,
        ):
    '''
    TEST PHASE
    '''
    LOGGER.info('Predict using ADALINE classifier')

    ## notation
    W, X = parameters, features
    N, D = X.shape
    LOGGER.debug('using weights: %s' % parameters)
    LOGGER.debug('on X: (%s, %s)' % (N, D))

    ## apply model to calculate output
    O = tanh(np.dot(X, W))
    LOGGER.info('weighted sum O: %s' % np.ravel(O))

    ## predict label using a threshold
    T = np.ones(O.shape)
    T[O < 0] = -1
    LOGGER.info('thresholded: %s' % np.ravel(T))

    return T
