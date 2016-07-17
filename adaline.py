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


def train_adaline(features, labels,
                  W=None,
                  rate=0.1,
                  termination_condition=max_iters(100),
                  label_type='01',
                  verbose=False):
    '''
    Returns the optimal weights for a given training set (features
    and corresponding label inputs) for the ADALINE model.
    These weights are found using the gradient descent method.
    /!\ Assumes bias term is already in the features input.

    TRAINING PHASE

    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and D:        the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (-1 and 1)
    - W:        D * 1 Numpy vector, beginning weights
    - rate:     learning rate, a float between 0 and 1
    - termination_condition: self-explanatory

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
    if W is None:
        W = np.zeros((D, 1))

    ## 2. Evaluate the termination condition
    i = 1
    while not termination_condition():

        ## current iteration classifier output
        O = np.dot(X, W)

        ## batch gradient descent
        gradient = -np.mean(np.multiply((Y - O), X), axis=0)
        gradient = gradient.reshape(W.shape)

        ## 3. Update weights
        W = W - rate * gradient

        ## Keep track of error and cost (weights from previous iteration)
        if label_type is '01':
            T = np.zeros(O.shape) # equivalent to threshold/step activation function
            T[O > 0.5] = 1
        else: # label type is '-11'
            T = np.ones(O.shape) # equivalent to threshold/step activation function
            T[O < 0] = -1

        current_error = get_error(T, Y)
        error.append(current_error)

        current_cost = get_cost(Y, O)
        cost.append(current_cost)

        if verbose: print('iteration %d:\tcost = %.3f' % (i, cost[-1]))
        i += 1

    return W, cost, error


def test_adaline(weights, features,
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
    X, W = features, weights
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

    ## calculate cost
    ## TODO ask why are we calculating this cost ?
    cost = get_cost(T, O)

    return labels, cost


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
    weights, cost, error = train_adaline(features=x, labels=y,
                                         rate=1,
                                         termination_condition=max_iters(100))
    print('\n   cost: %.3f' % cost[-1])
    print('\n  error: %.3f' % error[-1])
    print('\nweights: %.3f' % weights[0])
    #from code import interact; interact(local=dict(globals(), **locals()))
    for w in weights[1:]:
        print('         %.3f' % w)
    return

if __name__ == '__main__':
    sys.exit(main())

