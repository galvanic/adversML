# coding: utf-8

'''
Inputs:
- Y: true labels
- O: output, predicted labels by model

TODO could actually draw a table for TP, FP, FN, TN ? :)
'''
import numpy as np


def get_cost(Y, O):
    '''
    Calculate cost using Means Squared

    TODO ask does this make sense in here or grouped with gradient descent ?
    '''
    cost = np.mean(np.square(Y - O))
    return cost


def get_error(Y, O):
    '''
    Calculates mean error over samples
    '''
    error = np.mean(Y != O)
    return error


def get_FPR(Y, O, ham_label=0, spam_label=1):
    '''
    Calculates False Positive Rate (=fall-out), also called false alarm rate
    '''
    FP = np.sum((O == spam_label) & (Y == ham_label))
    N =  np.sum(Y == ham_label) ## FP + TN
    FPR = FP / N
    return FPR


def get_FNR(Y, O, ham_label=0, spam_label=1):
    '''
    Calculates False Negative Rate, also called miss rate
    '''
    FN = np.sum((O == ham_label) & (Y == spam_label))
    P =  np.sum(Y == spam_label) ## TP + FN
    FNR = FN / P
    return FNR


