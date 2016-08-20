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


def compute_output(X, W):
    '''output to train on'''
    return tanh(np.dot(X, W))

def compute_prediction(output, ham_label=-1):
    '''class label prediction from output'''
    O = output

    ## T is equivalent to threshold/step activation function
    if ham_label is -1:              ## spam label assumed 1
        T = np.sign(O)
    else:    ## ham label is assumed 0, spam label assumed 1
    ## T is equivalent to threshold/step activation function
        T = (O > 0.5)

    return T


@log
def fit(features, labels,
        ## params:
        gradient_descent_method,
        batch_size,
        max_epochs,
        divergence_threshold,
        learning_rate=0.5,
        initial_weights=None,
        convergence_threshold=1e-5,
        convergence_look_back=2,
        ham_label=-1,
        spam_label=1,
        ):
    '''
    Returns the optimal weights for a given training set (features
    and corresponding label inputs) for the logistic regression model.
    These weights are found using the gradient descent method.
    /!\ Assumes bias term is already in the features input.

    TRAINING PHASE

    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and  D: the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (-1 and 1)

    Output:
    - W: D * 1 Numpy vector of real values

    '''

    W = gradient_descent(features, labels,
        compute_output,
        predict,
        gradient_descent_method=gradient_descent_method,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        initial_weights=initial_weights,
        convergence_threshold=convergence_threshold,
        convergence_look_back=convergence_look_back,
        divergence_threshold=divergence_threshold,
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

    ## apply model to compute output
    O = compute_output(X, W)

    ## predict label using a threshold
    T = compute_prediction(O)

    return T
