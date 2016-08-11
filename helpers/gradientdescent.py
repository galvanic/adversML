# coding: utf-8
from __future__ import division
'''
'''
import logging
LOGGER = logging.getLogger(__name__)

import numpy as np
from helpers.performance import get_error


def gradient_descent(features, labels,
        calculate_output,
        predict,
        ## params:
        gradient_descent_method='stochastic',
        batch_size=10,
        initial_weights=None,
        learning_rate=0.05,
        max_epochs=1000,
        ):
    '''
    '''
    LOGGER.info('learning rate: %f' % learning_rate)
    LOGGER.info('using %s' % gradient_descent_method)


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

        ## mix up samples (they will therefore be fed in different order at
        ## each training) -> commonly accepted to improve gradient descent,
        ## making convergence faster)
        if epoch % N == 0:
            permuted_indices = np.random.permutation(N)

        ## TODO implement GD method as different function based on argument
        if gradient_descent_method == 'stochastic':
            ## stochastic GD: only update using 1 sample
            sample = permuted_indices[epoch % N]
            LOGGER.debug('- sample (%d/%d): %d' % (epoch % N, N, sample))
            x, y = X[sample].reshape(1, D), Y[sample]

        elif gradient_descent_method == 'mini-batch':
            start = (epoch * batch_size) % N
            end = (epoch * batch_size + batch_size) % N or None
            samples = permuted_indices[start:end]
            LOGGER.debug('- samples (%d-%s/%d): %s' % (start, end, N, samples))
            x, y = X[samples], Y[samples]

        elif gradient_descent_method == 'batch':
            x, y = X, Y

        else: ## batch
            x, y = X, Y

        LOGGER.debug('- x: (%s, %s)' % x.shape)
        LOGGER.debug('- y: (%s, %s)' % y.shape)

        ## classifier output of current epoch
        o = calculate_output(x, W)
        LOGGER.debug('- output: %s' % np.ravel(o))

        ## gradient descent: minimise the cost function
        ## gradient equation was obtained by deriving the LMS cost function
        gradient = -np.mean(np.multiply((y - o), x), axis=0)
        LOGGER.debug('- gradient: %s' % gradient)

        ## update weights
        W = W - learning_rate * gradient.reshape(W.shape)
        LOGGER.debug('- weights: %s' % np.ravel(W))

        ## Keep track of cost and error
        P = predict(W, X)
        error = get_error(Y, P)
        cost = get_cost(Y, P)
        LOGGER.info('- cost = %.2E' % cost)
        LOGGER.info('- error = %.2f' % error)

    return W




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


def get_cost(Y, O):
    '''Calculate cost using Means Squared'''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    cost = np.mean(np.square(Y - O))
    return cost

