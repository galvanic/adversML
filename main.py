#!/usr/bin/env python3
# coding: utf-8
from __future__ import division
'''
'''
import sys
import os
import time

from helpers.pipeline import perform_experiment_batch


def main(infolder, outfolder):

    ###
    ### SETUP EXPERIMENTS
    ###

    ## put iteration last, but other dimensions is preference only
    ## TODO use OrderedDict
    parameter_ranges = {
        'classifier': ['adaline', 'logistic regression', 'naive bayes'],
        'attack': ['empty', 'dictionary', 'none'],
        ('attack_parameters', 'percentage_samples_poisoned'): [0, .2, .5],
        'repetition': range(1, 20+1),
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
        'attack_parameters': {},
        'attack': None,
    }

    ###
    ### CARRY OUT EXPERIMENTS
    ###

    df = perform_experiment_batch(infolder, parameter_ranges, fixed_parameters)

    ###
    ### SAVE EXPERIMENT RESULTS
    ###

    saved_at = time.strftime('%y%m%d%H%M', time.localtime(time.time()))
    outfilepath = os.path.join(outfolder, saved_at)
    df.to_pickle('%s-df.dat' % outfilepath)

    ## save also as string for human readability
    ## change column and index names for clarity
    change_name = lambda old, new, lst: [new if (x == old) else x for x in lst]
    new_names = change_name('percentage_samples_poisoned', '% poisoned', df.index.names)
    df.index = df.index.set_names(new_names)

    with open('%s-df.txt' % outfilepath, 'w') as outfile:
        outfile.write(df.to_string(col_space=8, float_format='%.2f'))

    return df


if __name__ == '__main__':

    infolder = sys.argv[1] if len(sys.argv) > 1 else '../datasets/processed'
    outfolder = sys.argv[2] if len(sys.argv) > 2 else '.'
    sys.exit(main(infolder, outfolder))

