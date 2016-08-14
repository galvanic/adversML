# coding: utf-8
from __future__ import division
'''
'''
import os
import time
import pandas as pd

import logging
logger = logging.getLogger(__name__)


def get_time_id(time_format='%y%m%d%H%M'):
    '''
    '''

    time_id = time.strftime(time_format, time.localtime(time.time()))
    return time_id


def save_df(df, outfolder, experiment_id=None):
    '''
    '''

    saved_at = str(experiment_id) or get_time_id()
    outfilepath = os.path.join(outfolder, saved_at)
    df.to_pickle('%s-df.dat' % outfilepath)
    logger.info('Saved to %s\n' % outfilepath)

    ## save also as string for human readability
    string = df.to_string(col_space=8, float_format=lambda x: '%.2f' % x)
    logger.info('\n%s' % string)

    return


def join_repetitions(ifilepaths):
    '''
    '''

    dfs = [pd.read_pickle(filepath) for filepath in ifilepaths]
    from code import interact; interact(local=dict(locals(), **globals()))
    return df

