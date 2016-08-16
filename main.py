#!/usr/bin/env python3
# coding: utf-8
from __future__ import division
'''
'''
import sys
import yaml
import os.path

import logging.config
from helpers.logging import LOGGING_CONFIG

from helpers.batch import perform_experiment_batch
from helpers.i_o import get_time_id, save_df


def main(batch_specs_filepath, infolder, outfolder, num_threads=1):
    '''
    '''

    ## load data
    with open(batch_specs_filepath, 'r') as infile:
        batch_specs = yaml.safe_load(infile)

    fixed_parameters = batch_specs['fixed_parameters']
    parameter_ranges = batch_specs['parameter_ranges']

    batch_id = str(get_time_id())
    fixed_parameters['batch_id'] = batch_id

    log_filepath = os.path.join(outfolder, '%s.log' % batch_id)
    LOGGING_CONFIG['handlers']['file']['filename'] = log_filepath
    logging.config.dictConfig(LOGGING_CONFIG)

    ## carry out experiments
    df = perform_experiment_batch(parameter_ranges, fixed_parameters, infolder,
        num_threads=num_threads)

    ## save results
    save_df(df, outfolder, batch_id, batch_specs_filepath)

    return


if __name__ == '__main__':

    code_dir = os.path.dirname(os.path.realpath(__file__))
    top_dir = os.path.split(code_dir)[0]

    batch_specs_filepath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(code_dir, 'config', 'example.yaml')
    infolder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(top_dir, 'datasets', 'processed')
    outfolder = sys.argv[3] if len(sys.argv) > 3 else '.'
    num_threads = int(sys.argv[4]) if len(sys.argv) > 4 else None

    sys.exit(main(batch_specs_filepath, infolder, outfolder, num_threads=num_threads))

