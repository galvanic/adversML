#!/usr/bin/env python3
# coding: utf-8
from __future__ import division
'''
'''
import sys
import os
import time
import yaml

from helpers.pipeline import perform_experiment_batch


def save_df(df, outfilepath):
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


def main(parameter_ranges_filepath, infolder, outfolder,
        fixed_parameters_filepath='./default_spec.yaml',
        num_threads=8):
    '''
    '''

    with open(fixed_parameters_filepath, 'r') as infile:
        fixed_parameters = yaml.load(infile)

    with open(parameter_ranges_filepath, 'r') as infile:
        parameter_ranges = yaml.load(infile)

    df = perform_experiment_batch(parameter_ranges, fixed_parameters, infolder,
        use_threads=True, num_threads=num_threads)
    save_df(df, outfolder)

    return


if __name__ == '__main__':

    parameter_ranges_filepath = sys.argv[1] if len(sys.argv) > 1 else './example.yaml'
    infolder = sys.argv[2] if len(sys.argv) > 2 else '../datasets/processed'
    outfolder = sys.argv[3] if len(sys.argv) > 3 else '.'
    num_threads = int(sys.argv[4]) if len(sys.argv) > 4 else None

    sys.exit(main(parameter_ranges_filepath, infolder, outfolder, num_threads=num_threads))

