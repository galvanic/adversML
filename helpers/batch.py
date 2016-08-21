# coding: utf-8
from __future__ import division
'''
'''
import logging
logger = logging.getLogger(__name__)
from pprint import pformat

import sys
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool

from helpers.specs import generate_specs


def run_experiment_batch(parameter_ranges, default_parameters, run_experiment,
        num_threads=1):
    '''

    Inputs:
    - parameter_ranges
    - default_parameters
    - run_experiment: function

    Outputs:
    - df: pandas Dataframe of experiment metrics
    '''
    logger.info('Default parameters:\n%s\n' % pformat(default_parameters))
    logger.info('Experiment ranges:\n%s\n' % pformat(parameter_ranges))

    ## extract names to use later for DF
    try:
        dimension_names, keys, variations= zip(*((p['name'], tuple(p['key']), p['values']) for p in parameter_ranges))
    except TypeError as e:
        sys.stderr.write('No range of parameters specified\n')
        sys.exit(1)

    ## get all possible variations for specs
    specifications = generate_specs(zip(keys, variations), default_parameters)
    logger.info('Amount of specifications: %d' % len(specifications))

    ## run each experiment
    use_threads = bool(num_threads)
    if use_threads:
        logger.info('Using %s threads' % num_threads)
        with ThreadPool(processes=num_threads) as pool:
            results = pool.map(run_experiment, specifications)
    else:
        results = map(run_experiment, specifications)

    ## put into DataFrame for analysis
    idx = pd.MultiIndex.from_product(variations, names=dimension_names)
    df = pd.DataFrame.from_records(data=results, index=idx)
    df.columns.names = ['metrics']

    df['experiment_id'] = [spec['experiment_id'] for spec in specifications]
    df = df.set_index('experiment_id', append=True)
    #df = df.reorder_levels(('experiment_id',) + dimension_names)

    return df

