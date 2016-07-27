# coding: utf-8
from __future__ import division
'''
Implementation of logistic regression
Adapted from: https://github.com/kaylashapiro/SpamFilter/blob/master/Code/logisticReg.py
'''
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
        num_epochs=5,
        termination_condition=None,
        ham_label=-1,
        spam_label=1,
        verbose=True,
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
    if not termination_condition:
        termination_condition = max_iters(num_epochs)

    ## notation
    X, Y = features, labels
    N, D = X.shape           # N #training samples; D #features

    ## initialise weights
    W = np.zeros((D, 1)) if initial_weights is None else initial_weights.reshape((D, 1))


    for epoch in range(num_epochs):

        permutated_indices = np.random.permutation(N)
        X = X[permutated_indices]
        Y = Y[permutated_indices]

        ## stochastic gradient descent, sample by sample
        for sample in range(N):
            x, y = X[sample], Y[sample]

            ## classifier output of current epoch
            O = tanh(np.dot(x, W))

            ## gradient descent: calculate gradient
            gradient = np.multiply((O - y), x)

            ## update weights
            W = W - learning_rate * gradient.reshape(W.shape)

        P = predict(W, X)
        error = get_error(Y, P)
        cost = get_cost(Y, P)
        if verbose: print('epoch %d:\tcost = %.3f\terror = %.3f' % (epoch, cost, error))

    return W


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

    ## apply model to calculate output
    O = tanh(np.dot(X, W))

    ## predict label using a threshold
    T = np.ones(O.shape)
    T[O < 0] = -1

    return T
