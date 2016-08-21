#!/usr/bin/env python3
# coding: utf-8
from __future__ import division
'''
'''
import sys
import yaml
import os.path

import logging.config
from helpers.logging import LOGGING_CONFIG, COMMIT_HASH

from helpers.batch import run_experiment_batch
from helpers.i_o import get_time_id, save_df

## experiment types
from pipelines import offline as OfflineTraining
from pipelines import adaptive as AdaptiveCombination

experiment_function = {
    'offline training': OfflineTraining.run_experiment,
    'adaptive combination': AdaptiveCombination.run_experiment,
}


def main(batch_specs_filepath, infolder, outfolder, num_threads=1):
    '''
    '''

    ## load config for experiments to run
    with open(batch_specs_filepath, 'r') as infile:
        batch_specs = yaml.safe_load(infile)

    default_parameters = batch_specs['default_parameters']
    parameter_ranges = batch_specs['parameter_ranges']

    ## set up parameters that are experiment batch specific
    batch_id = str(get_time_id())
    default_parameters['batch_id'] = batch_id
    default_parameters['commit_hash'] = COMMIT_HASH
    default_parameters['dataset_dirpath'] = infolder ## should this just be in config ?

    log_filepath = os.path.join(outfolder, '%s.log' % batch_id)
    LOGGING_CONFIG['handlers']['file']['filename'] = log_filepath
    logging.config.dictConfig(LOGGING_CONFIG)

    ## run experiments
    df = run_experiment_batch(parameter_ranges, default_parameters,
        run_experiment=experiment_function[default_parameters['experiment']],
        num_threads=num_threads)

    ## save results
    save_df(df, outfolder, batch_id, batch_specs_filepath)

    return


if __name__ == '__main__':

    code_dir = os.path.dirname(os.path.realpath(__file__))
    top_dir = os.path.split(code_dir)[0]

    batch_specs_filepath = sys.argv[1]
    infolder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(top_dir, 'datasets', 'processed')
    outfolder = sys.argv[3] if len(sys.argv) > 3 else '.'
    num_threads = int(sys.argv[4]) if len(sys.argv) > 4 else None

    sys.exit(main(batch_specs_filepath, infolder, outfolder, num_threads=num_threads))

