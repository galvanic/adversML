#!/usr/bin/env python
# coding: utf-8

'''
Translation of Luis Munoz's MATLAB code for the Naive Bayes classifier model.
Not tested yet.
'''
import sys


def NaiveBayesBinaryClassifier(X_train, X_test, Y_train, Y_test):
    '''
    Computes the training and test error of a Naive Bayes binary
    classifier for features following a Bernouilli distribution.
    For example, a feature could be the presence of the word
    'viagra' or not (1 or 0 respectively).

    Uses Maximum Likelihood.

    There are D features and N samples. N_train is the number of
    training samples.

    Input:
    - X_train: N_train * D matrix of features for training
               Features are binary (1 or 0)
    - Y_train: N_train * 1 vector of labels for each train sample
    - X_test:  N_test  * D matrix of features for testing
    - Y_test:  N_test  * 1 vector of labels for each test example

    Output:
    - error_train: classification error for the training data
    - error_test:  classification error for the test data
    - pfa:         false alarm probability on the test data
    - pm:          miss probability on the test data
    - posteriors:  values of the posterior of the parameters

    TODO ASK which parameters ?
    '''
    N_train, D = X_train.shape # number of N:training samples, D:features
    N_test = X_test.shape[0]

    # prior probability based on the training samples
    prior = np.sum(Y_train)/N_train

    # parameters of the model (for the classes 1 and 0 respectively)
    p1 = np.zeros((D, 1))
    p0 = np.zeros((D, 0))

    # indices of the training samples labeled as 1 and 0 respectively
    ind1 = Y_train[Y_train == 1]
    ind0 = Y_train[Y_train == 0]

    # tolerance factor (useful when the parameters take values of 0 or 1)
    tolerance = 1e-50

    # estimate the parameters of the likelihood for class 1
    p1 = np.sum(X_train[ind1, :]).T / len(ind1)
    c = np.arange(len(p1))[np.isnan(p1)]
    p1[c] = tolerance
    c = p1[p1 == 1]
    p1[c] = 1 - tolerance
    c = p1[p1 == 0]
    p1[c] = tolerance

    # estimate the parameters of the likelihood for class 0
    p0 = np.sum(X_train[ind0, :]).T / len(ind0)
    c = np.arange(len(p0))[np.isnan(p0)]
    p0[c] = tolerance
    c = p0[p0 == 1]
    p0[c] = 1 - tolerance
    c = p0[p0 == 0]
    p0[c] = tolerance

    ## logs are used because otherwise multiplications of very small numbers

    # compute train predictions
    predictions_train = np.zeros((N_train, 1))
    for ii in range(N_train):
        log_post1 = np.log(prior) + X_train[ii, :]*np.log(p1) + (1-X_train[ii, :])*np.log(1-p1)
        log_post0 = np.log(1-prior) + X_train[ii, :]*np.log(p0) + (1-X_train[ii, :])*np.log(1-p0)
        predictions_train[ii] = (log_post1 > log_post0)

    # compute test predictions
    predictions_test = np.zeros((N_test, 1))
    for ii in range(N_test):
        log_post1 = np.log(prior) + X_test[ii, :]*np.log(p1) + (1-X_test[ii, :])*np.log(1-p1)
        log_post0 = np.log(1-prior) + X_test[ii, :]*np.log(p0) + (1-X_test[ii, :])*np.log(1-p0)
        predictions_test[ii] = (log_post1 > log_post0)

    # posterior parameters
    posteriors = { 'p0': p0; 'p1': p1 }

    ## monitoring of performance
    # compute train and test errors
    error_train = np.sum(predictions_train != Y_train) / N_train
    error_test = np.sum(predictions_test != Y_test) / N_test

    # false alarm probability in the test data
    pfa = np.sum((predictions_test == 1) & (Y_test == 0)) / np.sum(Y_test == 0)

    # miss probability in the test data
    pm = np.sum((predictions_test == 0) & (Y_test == 1)) / np.sum(Y_test == 1)

    return error_train, error_test, pfa, pm, posteriors


def main():
    return


if __name__ == '__main__':
    sys.exit(main())
