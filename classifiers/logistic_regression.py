# coding: utf-8
from __future__ import division
'''
Implementation of logistic regression
Adapted from: https://github.com/kaylashapiro/SpamFilter/blob/master/Code/logisticReg.py
'''
import numpy as np
from helpers.logging import tls, log
from helpers.gradientdescent import gradient_descent


## helpers
sigmoid = lambda z: 1 / (1 + np.exp(-z))
tanh = lambda z: np.tanh(z)


def calculate_output(X, W):
    '''output to train on'''
    return tanh(np.dot(X, W))


@log
def fit(features, labels,
        ## params:
        gradient_descent_method='stochastic',
        initial_weights=None,
        learning_rate=0.5,
        max_epochs=2,
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

    W = gradient_descent(features, labels,
        calculate_output,
        predict,
        gradient_descent_method=gradient_descent_method,
        initial_weights=initial_weights,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        )

    return W


@log
def predict(parameters, features,
        ## params
        ham_label=-1,
        spam_label=1,
        ):
    '''
    TEST PHASE
    '''

    ## notation
    W, X = parameters, features
    N, D = X.shape
    tls.logger.debug('using weights: %s' % parameters)
    tls.logger.debug('on X: (%s, %s)' % (N, D))

    ## apply model to calculate output
    O = calculate_output(X, W)
    tls.logger.debug('weighted sum O: %s' % np.ravel(O))

    ## predict label using a threshold
    T = np.ones(O.shape)
    T[O < 0] = -1
    tls.logger.debug('thresholded: %s' % np.ravel(T))

    return T
