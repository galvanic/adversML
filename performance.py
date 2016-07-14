# coding: utf-8

'''
Inputs:
- Y: true labels
- O: output, predicted labels by model

TODO could actually draw a table for TP, FP, FN, TN ? :)
'''
import numpy as np


def cost(Y, O):
    '''
    Calculate cost using Means Squared

    TODO ask does this make sense in here or grouped with gradient descent ?
    '''
    cost = np.mean(np.square(T - O))
    return cost


def error(Y, O):
    '''
    Calculates error
    '''
    N = len(Y)
    error = np.sum(O != Y) / N
    return error


def FPR(Y, O, ham_label=0, spam_label=1):
    '''
    Calculates False Positive Rate (=fall-out), also called false alarm rate
    '''
    FP = np.sum((O_test == spam_label) & (Y_test == ham_label))
    N =  np.sum(Y_test == ham_label) ## FP + TN
    FPR = FP / N
    return FPR


def FNR(Y, O, ham_label=0, spam_label=1):
    '''
    Calculates False Negative Rate, also called miss rate
    '''
    FN = np.sum((O_test == ham_label) & (Y_test == spam_label))
    P =  np.sum(Y_test == spam_label) ## TP + FN
    FNR = FN / P
    return FNR


