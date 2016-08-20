# coding: utf-8
from __future__ import division
'''
Implementation of the Adaline model.
Training is done using batch gradient descent.

TODO ? implement regularisation
TODO ? cost and error could be measured outside the function
     or at least use a callable to compute them, otherwise duplicated code
     across models
TODO clean up the code further, especially duplicated sections (adaline model
     etc.)
'''
import numpy as np
from helpers.gradientdescent import gradient_descent
from helpers.logging import tls, log


def compute_output(X, W):
    '''output to train on'''
    ## specialty of ADALINE is that training is done on the weighted sum,
    ## _before_ the activation function
    return np.dot(X, W)

def compute_prediction(output, ham_label=-1):
    '''class label prediction from output'''
    O = output

    ## T is equivalent to threshold/step activation function
    if ham_label is -1:              ## spam label assumed 1
        T = np.sign(O)
    else:    ## ham label is assumed 0, spam label assumed 1
        T = (O > 0.5)

    return T


@log
def fit(features, labels,
        ## params:
        gradient_descent_method,
        batch_size,
        max_epochs,
        divergence_threshold,
        learning_rate=0.1,
        initial_weights=None,
        convergence_threshold=1e-5,
        convergence_look_back=2,
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
    X, W = features, parameters

    ## apply model
    O = compute_output(X, W)

    ## compute predicted class labels
    T = compute_prediction(O)

    return T

