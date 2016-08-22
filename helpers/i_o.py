# coding: utf-8
from __future__ import division
'''
'''
import os
import time
import shutil
import pandas as pd

import logging
logger = logging.getLogger(__name__)


def get_time_id(time_format='%y%m%d%H%M'):
    '''
    '''

    time_id = time.strftime(time_format, time.localtime(time.time()))
    return time_id


def save_df(df, outfolder, experiment_id=None, specs_filepath=None):
    '''
    '''

    saved_at = str(experiment_id) or get_time_id()
    outfilepath = os.path.join(outfolder, saved_at)
    df.to_pickle('%s-df.dat' % outfilepath)
    logger.info('Saved to %s\n' % outfilepath)

    if specs_filepath:
        experiment_purpose = os.path.basename(specs_filepath)[:-5]
        shutil.move(specs_filepath, '%s-%s.yaml' % (outfilepath, experiment_purpose))

    ## save also as string for human readability
    if len(df.columns) > 8: ## TODO should be individual to ExperimentBatchSetup class or something
        df = df.xs(df.columns.levels[1][-2], axis=1, level='timestep')
    string = df.to_string(col_space=8, float_format=lambda x: '%.3f' % x)
    logger.info('\n%s' % string)

    return


def join_repetitions(ifilepaths):
    '''
    Assumes:
    - each individual df to be merged has an experiment_id index column
    - indices are the same for all dfs

    TODO use keys argument to pd.concat with value range(num_iterations)
         and then reorder index levels to put iteration at the end
         eg: pd.concat(dfs, keys=range(len(dfs)))
    '''

    dfs = [pd.read_pickle(filepath) for filepath in ifilepaths]
    df = pd.concat(dfs).sort_index()

    return df

