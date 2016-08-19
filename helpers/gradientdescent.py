# coding: utf-8
from __future__ import division
'''
'''
import numpy as np
import math
from collections import deque
from functools import partial

from helpers.logging import tls, log
from helpers.performance import get_error


def get_cost(Y, O):
    '''Calculate cost using Means Squared'''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    cost = np.mean(np.square(Y - O))
    return cost


@log
def get_mini_batch(X, Y, permuted_indices, batch_ii, batch_size):
    '''
    batch_ii is the current iteration of batch in the whole training set
    TODO to avoid keeping track of batch_ii, this could nicely be refactored
         into a generator
    '''
    N, D = X.shape

    start = (batch_ii * batch_size) % N
    end = (batch_ii * batch_size + batch_size) % N or None
    end = end if (end and end > start) else None
    samples = permuted_indices[start:end]
    tls.logger.debug('  samples (%d-%s/%d): %s' % (start, end, N, samples))

    x, y = X[samples], Y[samples]
    return x, y


@log
def gradient_descent(features, labels,
        ## functions specific to classifier:
        calculate_output,
        predict,
        ## params:
        gradient_descent_method,
        batch_size,
        learning_rate,
        max_epochs,
        initial_weights,
        convergence_threshold,
        convergence_look_back,
        divergence_threshold,
        ):
    '''
    Returns the optimal weights for a given training set (features
    and corresponding label inputs) for the given model*
    These weights are found using the gradient descent method.
    /!\ Assumes bias term is already in the features input.

    *The given model is determined by the `calculate_output` and
    `predict` functions.

    TRAINING PHASE


    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and  D: the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (-1 and 1)
    - gradient_descent_method: string
    - batch_size:              int, between 1 and N
    - learning_rate:           float, between 0 and 1
    - max_epochs:              int, >= 0
    - initial_weights:         D * 1 Numpy vector
    - convergence_threshold:   float, very small number
    - convergence_look_back:   int, >= 1
        stops if the error difference hasn't been over threshold
        for the last X epochs

    Output:
    - W: D * 1 Numpy vector of real values


    TODO adaptive learning rate ?
    TODO yield cost, error, weights as it is learning ?
         this could allow possibility to inject new learning rate during
         training
    TODO determine what to do if cost is rising instead of falling:
         - abort ?
         - retry w lower learning rate ?
    TODO output is calculated twice per iteration (calculate_output and predict)
         refactor
    '''
    tls.logger.debug('learning rate: %f' % learning_rate)
    tls.logger.debug('using %s' % gradient_descent_method)

    ## notation
    X, Y = features, labels
    N, D = X.shape           # N #training samples; D #features
    tls.logger.debug('X: (%s, %s)\tY: %s' % (N, D, str(Y.shape)))

    batch_size_per_method = {
        'stochastic': 1,
        'mini-batch': batch_size,
        'batch': N,
    }
    batch_size = batch_size_per_method[gradient_descent_method]
    get_batch = partial(get_mini_batch, batch_size=batch_size)

    ## initialise weights
    W = np.zeros((D, 1)) if initial_weights is None else initial_weights.reshape((D, 1))

    ## evaluate the termination condition
    previous_errors = deque(maxlen=convergence_look_back)
    previous_errors.append(0)

    for epoch in range(max_epochs): ## epoch defined as one pass through the whole training set
        tls.logger.info('epoch %d:' % epoch)

        ## mix up samples (they will therefore be fed in different order at
        ## each training) -> commonly accepted to improve gradient descent,
        ## making convergence faster)
        permuted_indices = np.random.permutation(N)

        total_batches = math.ceil(N / batch_size)  ## trains on all samples
        for batch_ii in range(total_batches):

            weight_update_iteration = epoch * total_batches + batch_ii
            tls.logger.debug('  weight update cycle %d' % weight_update_iteration)

            x, y = get_batch(X, Y, permuted_indices, batch_ii) ## also called pattern
                                                               ## (ie. "training pattern")
            tls.logger.debug('  x: (%s, %s)' % x.shape)
            tls.logger.debug('  y: %s' % str(y.shape))

            ## classifier output of current epoch
            o = calculate_output(x, W)

            ## gradient descent: minimise the cost function
            ## gradient equation was obtained by deriving the LMS cost function
            gradient = -np.mean(np.multiply((y - o), x), axis=0)

            ## update weights
            num_x = x.shape[0]
            update_coef = num_x / batch_size if num_x < batch_size else 1 ## weigh update relative
                                                                          ## to expected batch size
            if update_coef != 1: tls.logger.debug('  update coef: %.2f' % update_coef)
            W = W - learning_rate * gradient.reshape(W.shape) * update_coef

            ## Keep track of cost and error
            P = predict(W, X)
            cost = get_cost(Y, P)
            error = get_error(Y, P)
            tls.logger.debug('  cost = %.2e' % cost)
            tls.logger.debug('  error = %.2f' % error)

            ## check for convergence in last x weight update cycles
            if all(abs(np.array(previous_errors) - error) < convergence_threshold):
                tls.logger.info('converged')
                return W

            ## check for divergence TODO case when it oscillates
            if all(abs(np.array(previous_errors)[-2:] - error) > divergence_threshold):
                tls.logger.info('diverged')
                return W

            previous_errors.append(error)

    return W


