# coding: utf-8
from __future__ import division

'''
Implementation of the Adaline model.
Training is done using batch gradient descent.

TODO stochastic gradient descent (for online learning)
TODO ? make an Adaline class with train and test as methods
TODO ? implement regularisation
TODO ? cost and error could be measured outside the function
     or at least use a callable to calculate them, otherwise duplicated code
     across models
TODO clean up the code further, especially duplicated sections (adaline model
     etc.)
'''
import logging
import numpy as np

from helpers.gradientdescent import max_iters, get_cost
from helpers.performance import get_error


def fit(features, labels,
        ## params:
        initial_weights=None,
        learning_rate=0.05,
        max_epochs=200,
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
    logging.info('Training Adaline classifier')

    ## notation
    X, Y = features, labels
    N, D = features.shape   # N #training samples; D #features
    logging.debug('X: (%s, %s)\tY: (%s, %s)' % (N, D, *Y.shape))

    ## 1. Initialise weights
    W = np.zeros((D, 1)) if initial_weights is None else initial_weights.reshape((D, 1))
    logging.debug('W: %s' % W)

    ## 2. Evaluate the termination condition
    for epoch in range(max_epochs):
        logging.info('epoch %d:' % epoch)

        ## current iteration classifier output
        O = np.dot(X, W)
        logging.debug('- output: %s' % O)

        ## specialty of ADALINE is that training is done on the weighted sum,
        ## _before_ the activation function
        ## batch gradient descent
        gradient = -np.mean(np.multiply((Y - O), X), axis=0)
        logging.debug('- gradient" %s' % gradient)

        ## 3. Update weights
        W = W - learning_rate * gradient.reshape(W.shape)
        logging.debug('- weights: %s' % W)

        ## Keep track of error and cost (weights from previous iteration)
        ## T is equivalent to threshold/step activation function
        if ham_label is 0:               ## spam label assumed 1
            T = np.zeros(O.shape)
            T[O > 0.5] = 1
        else:   ## ham label is assumed -1, spam label assumed 1
            T = np.ones(O.shape)
            T[O < 0] = -1
        logging.debug('- activated output: %s' % T)

        error = get_error(Y, T)
        cost = get_cost(Y, O)
        logging.info('- cost = %.3f\terror = %.3f' % (cost, error))

    return W


def predict(parameters, features,
        ## params
        ham_label=-1,
        spam_label=1,
        ):
    '''
    TEST PHASE
    '''
    logging.info('Predict using ADALINE classifier')

    ## notation
    X, W = features, parameters
    N, D = features.shape
    logging.debug('using weights: %s' % parameters)
    logging.debug('on X: (%s, %s)' % (N, D))

    ## apply model
    O = np.dot(X, W)
    logging.info('weighted sum O: %s' % O)

    ## calculate output
    ## T is equivalent to threshold/step activation function
    if ham_label is 0:               ## spam label assumed 1
        T = np.zeros(O.shape)
        T[O > 0.5] = 1
    else:   ## ham label is assumed -1, spam label assumed 1
        T = np.ones(O.shape)
        T[O < 0] = -1
    logging.info('threshold T: %s' % T)

    return T

