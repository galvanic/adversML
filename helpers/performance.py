# coding: utf-8
from __future__ import division

'''
Inputs:
- Y: true labels
- O: output, predicted labels by model

TODO could actually draw a table for TP, FP, FN, TN ? :)
'''
import numpy as np
from sklearn.metrics import auc, roc_auc_score


def get_cost(Y, O):
    '''
    Calculate cost using Means Squared

    TODO ask does this make sense in here or grouped with gradient descent ?
    '''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    cost = np.mean(np.square(Y - O))
    return cost


def get_error(Y, O):
    '''
    Calculates mean error over samples
    '''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    error = np.mean(Y != O)
    return error


def get_FPR(Y, O, ham_label, spam_label):
    '''
    Calculates False Positive Rate (=fall-out), also called false alarm rate
    '''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    FP = np.sum((O == spam_label) & (Y == ham_label))
    N =  np.sum(Y == ham_label) ## FP + TN
    FPR = FP / N
    return FPR


def get_FNR(Y, O, ham_label, spam_label):
    '''
    Calculates False Negative Rate, also called miss rate
    '''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    FN = np.sum((O == ham_label) & (Y == spam_label))
    P =  np.sum(Y == spam_label) ## TP + FN
    FNR = FN / P
    return FNR


def get_TPR(Y, O, ham_label, spam_label):
    '''
    Calculates True Positive Rate
    '''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    TP = np.sum((O == spam_label) & (Y == spam_label))
    P =  np.sum(Y == spam_label) ## TP + FN
    TNR = TP / P
    return TNR


def get_AUC(Y, O, ham_label, spam_label):
    '''
    Calculates Area Under Curve of the ROC.
    '''
    AUC = roc_auc_score(Y, O)
    return AUC

