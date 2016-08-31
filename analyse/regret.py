# coding: utf-8
from __future__ import division
'''
df_regret, df_cumloss = compute_regret(df, classifier_names, 500)
'''
import pandas as pd
import numpy as np


CLASSIFIERS = ['fast', 'slow', 'combination']


def get_cumulative_loss(df, T, axis=0):
    '''Returns Series of cumulative loss over timesteps for given classifier loss'''
    column = df['loss'].rolling(window=T, axis=axis).sum()
    column = column.rename('cumulative loss') if axis == 0 else column
    return column


def compute_regret(DF, window_size, classifier_names=CLASSIFIERS):
    '''Returns (Series of regret values over timesteps,
                DataFrame of cumulative loss over timesteps)'''
    dfs = (get_cumulative_loss(DF[name], window_size) for name in classifier_names)
    df = pd.concat(dfs, axis=1, keys=classifier_names)

    ## calculate regret
    df_classifiers = df.drop('combination', axis=1)
    df_regret = (df['combination'] - df_classifiers.min(axis=1))
    df_regret = df_regret.rename(('regret', ''))

    ## keep cumulative loss
    ## TODO there must be a cleaner way to add a column level
    labels = [range(len(df.columns)), [0,]*len(df.columns)]
    idx = pd.MultiIndex(levels=[df.columns, ['cumulative loss']], labels=labels)
    df.columns = idx

    return df_regret, df


def transform_for_plot(df, T):
    ''''''
    df_regret, df_cumloss = compute_regret(df, T)

    df = df.join(df_regret)
    df = df.join(df_cumloss).sort_index(axis=1)
    return df


