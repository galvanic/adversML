# coding: utf-8
from __future__ import division
'''
Implementation of logistic regression
Adapted from: https://github.com/kaylashapiro/SpamFilter/blob/master/Code/logisticReg.py
'''
import numpy as np

from helpers.gradientdescent import max_iters
from helpers.performance import get_error

## helpers
sigmoid = lambda z: 1 / 1 + np.exp(-z)


def train(features, labels,
        ## params:
        initial_weights=None,
        learning_rate=0.01,
        epochs=100,
        termination_condition=None,
        threshold=1e-5,
        ham_label=-1,
        spam_label=1,
        verbose=False,
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
        termination_condition = max_iters(100)
    num_epochs = 100

    ## notation
    X, Y = features, labels
    N, D = X.shape           # N #training samples; D #features

    epoch_errors = np.zeros((epochs, 1))
    last_epoch_error = 1e6

    ## initialise weights
    W = np.zeros((D, 1)) if initial_weights is None else initial_weights.reshape((D, 1))

    permutated_indices = np.random.permutation(N)

    for epoch in range(num_epochs):
        X = X[permutated_indices]
        Y = Y[permutated_indices]

        ## stochastic gradient descent, sample by sample
        for sample in range(N):
            x, y = X[sample], Y[sample]

            ## classifier output of current epoch
            O = sigmoid(np.dot(x, W))

            ## gradient descent: calculate gradient
            gradient = np.multiply((O - y), x)

            ## update weights
            W = W - learning_rate * gradient.reshape(W.shape)

            O = test(W, X)
            epoch_errors[epoch] = get_error(Y, O)

            if np.abs(last_epoch_error - epoch_errors[epoch]) < threshold:
                break
            last_epoch_error = epoch_errors[epoch]

    return W


def test(parameters, features,
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

    probs = sigmoid(np.dot(X, W))
    predictions = np.ones((N, 1))
    predictions[probs < 0] = -1

    return predictions
