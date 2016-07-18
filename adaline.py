#!/usr/bin/env python3
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
import sys
import numpy as np

from performance import get_cost, get_error
from gradientdescent import max_iters


def train(features, labels,
        ## params:
        initial_weights=None,
        learning_rate=0.01,
        termination_condition=max_iters(100),
        ham_label=0,
        spam_label=1,
        verbose=False,
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
    - termination_condition: returns a bool

    Output:
    - W: D * 1 Numpy vector of real values

    TODO yield cost, error, weights as it is learning ?
         this could allow possibility to inject new learning rate during
         training
    TODO implement an autostop if cost is rising instead of falling ?
    '''
    ## 0. Prepare notations
    X, Y = features, labels
    N, D = features.shape   # N #training samples; D #features
    cost = []               # keep track of cost
    error = []              # keep track of error

    ## 1. Initialise weights
    W = np.zeros((D, 1)) if initial_weights is None else initial_weights.reshape((D, 1))

    ## 2. Evaluate the termination condition
    i = 1
    while not termination_condition():

        ## current iteration classifier output
        O = np.dot(X, W)

        ## specialty of ADALINE is that training is done on the weighted sum,
        ## _before_ the activation function
        ## batch gradient descent
        gradient = -np.mean(np.multiply((Y - O), X), axis=0)
        gradient = gradient.reshape(W.shape)

        ## 3. Update weights
        W = W - learning_rate * gradient

        ## Keep track of error and cost (weights from previous iteration)
        ## T is equivalent to threshold/step activation function
        if ham_label is 0:               ## spam label assumed 1
            T = np.zeros(O.shape)
            T[O > 0.5] = 1
        else:   ## ham label is assumed -1, spam label assumed 1
            T = np.ones(O.shape)
            T[O < 0] = -1

        current_error = get_error(T, Y)
        error.append(current_error)

        current_cost = get_cost(Y, O)
        cost.append(current_cost)

        if verbose: print('iteration %d:\tcost = %.3f' % (i, cost[-1]))
        i += 1

    return W#, cost, error


def test(parameters, features,
        ## params
        ham_label=0,
        spam_label=1,
        ):
    '''
    TEST PHASE

    TODO not sure what makes sense to measure here ?
         => performance can be calculated outside the function
    '''
    ## notation
    X, W = features, parameters
    N, D = features.shape

    ## apply model
    O = np.dot(X, W)

    ## calculate output
    ## T is equivalent to threshold/step activation function
    if ham_label is 0:               ## spam label assumed 1
        T = np.zeros(O.shape)
        T[O > 0.5] = 1
    else:   ## ham label is assumed -1, spam label assumed 1
        T = np.ones(O.shape)
        T[O < 0] = -1

    return T


def main():
    '''Test Adaline training'''
    ## dummy data
    ## 10 training samples and 3 features
    ## so a 10 * 3 matrix
    x = np.array([[1, 0, 1],
                  [0, 0, 0],
                  [1, 0, 1],
                  [1, 1, 1],
                  [1, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0]],
                  dtype=np.int8)
    y = np.array([[1],
                  [1],
                  [1],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [1]],
                  dtype=np.int8) #* 2 - 1

    ## train model
    weights = train(features=x, labels=y,
        learning_rate=1,
        termination_condition=max_iters(100))
    print('weights: %s' % ([' %.3f' % w for w in weights]))
    return

if __name__ == '__main__':
    sys.exit(main())

