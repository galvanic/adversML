# coding: utf-8
from __future__ import division
'''
Implementation of the Adaline model.
Training is done using batch gradient descent.

TODO ? implement regularisation
TODO ? cost and error could be measured outside the function
     or at least use a callable to calculate them, otherwise duplicated code
     across models
TODO clean up the code further, especially duplicated sections (adaline model
     etc.)
'''
import numpy as np
from helpers.gradientdescent import gradient_descent
from helpers.logging import tls, log


def calculate_output(X, W):
    '''output to train on'''
    ## specialty of ADALINE is that training is done on the weighted sum,
    ## _before_ the activation function
    return np.dot(X, W)


@log
def fit(features, labels,
        ## params:
        gradient_descent_method='stochastic',
        initial_weights=None,
        learning_rate=0.05,
        max_epochs=20,
        ham_label=-1,
        spam_label=1,
        ):
    '''
    Returns the optimal weights for a given training set (features
    and corresponding label inputs) for the ADALINE model.
    These weights are found using the gradient descent method.
    /!\ Assumes bias term is already in the features input.

    TRAINING PHASE

    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and  D: the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (-1 and 1)
    - initial_weights: D * 1 Numpy vector, beginning weights
    - learning_rate: learning rate, a float between 0 and 1

    Output:
    - W: D * 1 Numpy vector of real values

    TODO yield cost, error, weights as it is learning ?
         this could allow possibility to inject new learning rate during
         training
    TODO implement an autostop if cost is rising instead of falling ?
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
    X, W = features, parameters
    N, D = features.shape
    tls.logger.debug('using weights: %s' % np.ravel(W))
    tls.logger.debug('on X: (%s, %s)' % (N, D))

    ## apply model
    O = calculate_output(X, W)
    tls.logger.debug('weighted sum O: %s' % np.ravel(O))

    ## calculate predicted output
    ## T is equivalent to threshold/step activation function
    if ham_label is 0:               ## spam label assumed 1
        T = np.zeros(O.shape)
        T[O > 0.5] = 1
    else:   ## ham label is assumed -1, spam label assumed 1
        T = np.ones(O.shape)
        T[O < 0] = -1
    tls.logger.debug('- thresholded: %s' % np.ravel(T))

    return T

