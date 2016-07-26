#!/usr/bin/env python3
# coding: utf-8
from __future__ import division
'''
'''
import sys
import os
import time

from helpers.pipeline import perform_experiment_batch


def change_name(old, new, lst):
    names = [new if (x == old) else x for x in lst]
    return names


def main(outfolder):

    ## put iteration last, but other dimensions is preference only
    ## TODO use OrderedDict
    parameter_ranges = {
        'classifier': ['adaline', 'naive bayes'],
        'attack': ['ham', 'empty'],
        ('attack_parameters', 'percentage_samples_poisoned'): [0, .1, .2, .3, .4, .5],
        'iteration': range(1, 10+1),
    }

    fixed_parameters = {
        'dataset': 'trec2007',
        'dataset_filename': 'trec2007-1607201347',
        'label_type': {
            'ham_label': -1,
            'spam_label': 1,
        },
        'training_parameters': {},
        'testing_parameters': {},
    }

    df = perform_experiment_batch(parameter_ranges, fixed_parameters)

    ## save the data
    saved_at = time.strftime('%y%m%d%H%M', time.localtime(time.time()))
    outfilepath = os.path.join(outfolder, saved_at)
    df.to_pickle('%s-df.dat' % outfilepath)

    ## save also as string for human readability
    ## change column and index names for clarity
    new_names = change_name('percentage_samples_poisoned', '% poisoned', df.index.names)
    df.index = df.index.set_names(new_names)


    with open('%s-df.txt' % outfilepath, 'w') as outfile:
        outfile.write(df.to_string(col_space=8, float_format='%.3f'))

    return df


if __name__ == '__main__':

    outfolder = sys.argv[1] if len(sys.argv) > 1 else '.'
    sys.exit(main(outfolder))

