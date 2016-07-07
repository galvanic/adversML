#!/usr/bin/env python
# coding: utf-8

'''
'''
import sys
import numpy as np
import numpy.random as rand

### Helper functions for termination conditions

def Counter(max_iterations):
    i = max_iterations
    while True:
        yield i
        i -= 1

def max_iters(max_iterations):
    '''Assumes max_iterations is a Natural Integer.'''
    counter = Counter(max_iterations)
    return lambda: not next(counter)


def train_adaline(features, labels,
                  rate=0.1,
                  termination_condition=max_iters(100),
                  verbose=False):
    '''
    Returns the optimal weights for a given training set (features
    and corresponding label inputs) for the ADALINE model.
    These weights are found using the gradient descent method.
    /!\ Assumes bias term is already in the features input.

    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and D:        the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (0 and 1)
    - rate:     learning rate, a float between 0 and 1
    - termination_condition: self-explanatory

    Output:
    - optimal_weights: D * 1 Numpy vector of real values

    TODO implement stochastic gradient descent
    TODO implement regularization
    TODO yield learning rate as it is learning ?
    '''
    ## 0. Prepare notations
    X, Y = features, labels
    N, D = features.shape   # N #training samples; D #features
    cost = []               # keep track of cost
    error = []              # keep track of error

    ## 1. Initialise weights
    W = np.zeros((D, 1))

    ## 2. Evaluate the termination condition
    i = 1
    while not termination_condition():
        if verbose: print('iteration %d' % i)
        i += 1

        ## current iteration classifier output
        O = np.dot(X, W)

        ## batch gradient descent
        gradient = -1/N * np.sum(np.multiply((Y - O), X), axis=0)

        ## 3. Update weights
        W = W - rate * gradient.T

        ## Keep track of error and cost (weights from previous iteration)
        T = np.zeros(O.shape) # threshold/step activation function
        T[O > 0] = 1
        current_error = np.sum(T != Y)/N # mean error over samples
        error.append(current_error)

        current_cost = np.mean(np.square(Y-O)) # Means Square
        cost.append(current_cost)

    return W, cost, error


def main():
    '''Test Adaline training'''
    ## make dummy data
    ## 10 training samples and 3 features
    ## so a 10 * 3 matrix
    N = 10
    D = 3
    x = np.random.randint(2, size=(N,D))
    y = np.random.randint(2, size=(N,1))

    ## train model
    optimal_weights, cost, error = train_adaline(features=x, labels=y,
                                    rate=0.05,
                                    termination_condition=max_iters(10))
    print(cost)
    print(optimal_weights.T)
    return

if __name__ == '__main__':
    sys.exit(main())
