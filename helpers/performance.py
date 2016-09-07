# coding: utf-8
from __future__ import division

'''
Inputs:
- Y: true labels
- O: output, predicted labels by model

TODO could actually draw a table for TP, FP, FN, TN ? :)
'''
import numpy as np
from sklearn.metrics import auc, roc_auc_score, confusion_matrix


def get_error(Y, O):
    '''
    Calculates mean error over samples
    '''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    error = np.mean(Y != O)
    return error


def get_confusion_matrix(Y, O, ham_label, spam_label):
    '''
    '''
    cm = confusion_matrix(y_true=Y, y_pred=O, labels=(spam_label, ham_label))
    return cm


def get_TP(Y, O, ham_label, spam_label):
    '''
    '''
    TP = get_confusion_matrix(Y, O, ham_label, spam_label)[0][0]
    return TP

def get_TN(Y, O, ham_label, spam_label):
    '''
    '''
    TN = get_confusion_matrix(Y, O, ham_label, spam_label)[1][1]
    return TN

def get_FP(Y, O, ham_label, spam_label):
    '''
    '''
    FP = get_confusion_matrix(Y, O, ham_label, spam_label)[1][0]
    return FP

def get_FN(Y, O, ham_label, spam_label):
    '''
    '''
    FN = get_confusion_matrix(Y, O, ham_label, spam_label)[0][1]
    return FN


def get_FPR(Y, O, ham_label, spam_label):
    '''
    Calculates False Positive Rate (=fall-out), also called false alarm rate
    Y: true labels
    O: predicted labels
    '''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    FP = np.sum((O == spam_label) & (Y == ham_label))
    N =  np.sum(Y == ham_label) ## FP + TN
    FPR = FP / N
    return FPR


def get_FNR(Y, O, ham_label, spam_label):
    '''
    Calculates False Negative Rate, also called miss rate
    Y: true labels
    O: predicted labels
    '''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    FN = np.sum((O == ham_label) & (Y == spam_label))
    P =  np.sum(Y == spam_label) ## TP + FN
    FNR = FN / P
    return FNR


def get_TPR(Y, O, ham_label, spam_label):
    '''
    Calculates True Positive Rate
    Y: true labels
    O: predicted labels
    '''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    TP = np.sum((O == spam_label) & (Y == spam_label))
    P =  np.sum(Y == spam_label) ## TP + FN
    TNR = TP / P
    return TNR


def get_ROC_AUC(Y, O, ham_label, spam_label):
    '''
    Calculates Area Under Curve of the ROC.
    '''
    return roc_auc_score(Y, O)

