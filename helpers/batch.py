# coding: utf-8
from __future__ import division
'''
'''
import logging
logger = logging.getLogger(__name__)

import pandas as pd
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

from helpers.pipeline import perform_experiment
from helpers.specs import generate_specs


def perform_experiment_batch(parameter_ranges, fixed_parameters, infolder,
        use_threads=True, num_threads=8):
    '''

    Inputs:
    - infolder: path of directory where processed datasets are
    - parameter_ranges
    - fixed_parameters

    TODO what parts of the experiment specs are tied together ? and can
         therefore be simplified ?
    TODO make sure iteration is last ? or take it as seperate argument and put
         put it last ? OrderedDict should take in order of experiments, or the
         experiment_dimensions dictionary should already be an instance of
         Orditeration is last ? or take it as seperate argument and put
         put it last ? OrderedDict should take in order of experiments, or the
         experiment_dimensions dictionary should already be an instance of
         OrderedDict
    '''

    ## extract names to use later for DF
    dimension_names, keys, variations= zip(*((p['name'], tuple(p['key']), p['values']) for p in parameter_ranges))

    ## get all possible variations for specs
    specifications = generate_specs(zip(keys, variations), fixed_parameters)
    logger.info('Amount of specifications: %d' % len(specifications))

    ## perform each experiment (prepare function first)
    ## TODO incorporate infolder directly by changing the specs
    perform_exp = partial(perform_experiment, infolder=infolder)

    ## use threads
    if use_threads:
        logger.info('Using %s threads' % num_threads)
        with ThreadPool(processes=num_threads) as pool:
            results = pool.map(perform_exp, specifications)
    else:
        results = map(perform_exp, specifications)

    ## put into DataFrame for analysis
    idx = pd.MultiIndex.from_product(variations, names=dimension_names)
    df = pd.DataFrame.from_records(data=results, index=idx)
    df.columns.names = ['metrics']

    return df

