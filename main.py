#!/usr/bin/env python3
# coding: utf-8
from __future__ import division
'''
'''
import logging.config
import sys
import yaml
from pprint import pformat
import numpy as np
np.set_printoptions(precision=2, threshold=10e6, linewidth=10e10)

from helpers.pipeline import perform_experiment_batch
from helpers.i_o import get_time_id, save_df


def main(parameter_ranges_filepath, infolder, outfolder,
        fixed_parameters_filepath='config/default_spec.yaml',
        num_threads=8):
    '''
    '''
    batch_id = str(get_time_id())

    ## implement logging
    with open('config/logging.yaml', 'r') as infile:
        logging_config = yaml.load(infile)

    logging_config['handlers']['file']['filename'] = './%s.log' % batch_id
    logging.config.dictConfig(logging_config)


    ## load data
    with open(fixed_parameters_filepath, 'r') as infile:
        fixed_parameters = yaml.load(infile)

    fixed_parameters['experiment_batch_id'] = batch_id
    logging.info('Default parameters:\n%s\n' % pformat(fixed_parameters))

    with open(parameter_ranges_filepath, 'r') as infile:
        parameter_ranges = yaml.load(infile)
    logging.info('Experiment ranges:\n%s\n' % pformat(parameter_ranges))


    ## carry out experiments
    df = perform_experiment_batch(parameter_ranges, fixed_parameters, infolder,
        use_threads=True, num_threads=num_threads)

    ## save results
    save_df(df, outfolder, batch_id)

    return


if __name__ == '__main__':

    parameter_ranges_filepath = sys.argv[1] if len(sys.argv) > 1 else 'config/example.yaml'
    infolder = sys.argv[2] if len(sys.argv) > 2 else '../datasets/processed'
    outfolder = sys.argv[3] if len(sys.argv) > 3 else '.'
    num_threads = int(sys.argv[4]) if len(sys.argv) > 4 else None

    sys.exit(main(parameter_ranges_filepath, infolder, outfolder, num_threads=num_threads))

