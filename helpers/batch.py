# coding: utf-8
from __future__ import division
'''
'''
import logging
logger = logging.getLogger(__name__)

from pprint import pformat
import pandas as pd
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

from helpers.pipeline import perform_experiment
from helpers.specs import generate_specs


def perform_experiment_batch(parameter_ranges, fixed_parameters, infolder,
        num_threads=1):
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
    logger.info('Default parameters:\n%s\n' % pformat(fixed_parameters))
    logger.info('Experiment ranges:\n%s\n' % pformat(parameter_ranges))

    ## extract names to use later for DF
    dimension_names, keys, variations= zip(*((p['name'], tuple(p['key']), p['values']) for p in parameter_ranges))

    ## get all possible variations for specs
    specifications = generate_specs(zip(keys, variations), fixed_parameters)
    logger.info('Amount of specifications: %d' % len(specifications))

    ## perform each experiment (prepare function first)
    ## TODO incorporate infolder directly by changing the specs
    perform_exp = partial(perform_experiment, infolder=infolder)

    use_threads = bool(num_threads)
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

    df['experiment_id'] = [spec['experiment_id'] for spec in specifications]
    df = df.set_index('experiment_id', append=True)
    #df = df.reorder_levels(('experiment_id',) + dimension_names)

    return df

