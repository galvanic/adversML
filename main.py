#!/usr/bin/env python3
# coding: utf-8
from __future__ import division
'''
'''
import logging
import sys
import yaml
from pprint import pformat

from helpers.pipeline import perform_experiment_batch
from helpers.i_o import save_df




def main(parameter_ranges_filepath, infolder, outfolder,
        fixed_parameters_filepath='./default_spec.yaml',
        num_threads=8):
    '''
    '''
    ## load data

    with open(fixed_parameters_filepath, 'r') as infile:
        fixed_parameters = yaml.load(infile)

    with open(parameter_ranges_filepath, 'r') as infile:
        parameter_ranges = yaml.load(infile)

    logging.info('Loaded experiment specs:')
    logging.info('Default parameters:\n%s' % pformat(fixed_parameters))
    logging.info('Experiment ranges:\n%s' % pformat(parameter_ranges))

    ## carry out experiments
    df = perform_experiment_batch(parameter_ranges, fixed_parameters, infolder,
        use_threads=True, num_threads=num_threads)

    ## save results
    save_df(df, outfolder)

    return


if __name__ == '__main__':

    ## implement logging
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S')

    parameter_ranges_filepath = sys.argv[1] if len(sys.argv) > 1 else './example.yaml'
    infolder = sys.argv[2] if len(sys.argv) > 2 else '../datasets/processed'
    outfolder = sys.argv[3] if len(sys.argv) > 3 else '.'
    num_threads = int(sys.argv[4]) if len(sys.argv) > 4 else None

    sys.exit(main(parameter_ranges_filepath, infolder, outfolder, num_threads=num_threads))

