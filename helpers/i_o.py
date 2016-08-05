# coding: utf-8
from __future__ import division
'''
'''
import os
import time
import pandas as pd


def save_df(df, outfolder):
    '''
    '''
    saved_at = time.strftime('%y%m%d%H%M', time.localtime(time.time()))
    outfilepath = os.path.join(outfolder, saved_at)
    df.to_pickle('%s-df.dat' % outfilepath)

    ## save also as string for human readability
    with open('%s-df.txt' % outfilepath, 'w') as outfile:
        ## https://stackoverflow.com/questions/34097038/issue-calling-to-string-with-float-format-on-pandas-dataframe/34097171#34097171
        string = df.to_string(col_space=8, float_format=lambda x: '%.2f' % x)
        outfile.write(string)

    return


def join_repetitions(ifilepaths):
    '''
    '''
    dfs = [pd.read_pickle(filepath) for filepath in ifilepaths]
    from code import interact; interact(local=dict(locals(), **globals()))
    return df
