#!/usr/bin/env python3
# coding: utf-8
from __future__ import division
'''
'''
import sys
import os
import time

from helpers.pipeline import perform_experiment_batch


def save_df(df, outfilepath):
    '''
    '''
    saved_at = time.strftime('%y%m%d%H%M', time.localtime(time.time()))
    outfilepath = os.path.join(outfolder, saved_at)
    df.to_pickle('%s-df.dat' % outfilepath)

    ## save also as string for human readability
    with open('%s-df.txt' % outfilepath, 'w') as outfile:
        outfile.write(df.to_string(col_space=8, float_format='%.2f'))

    return


def main(infolder, outfolder):

    ###
    ### SETUP EXPERIMENTS
    ###

    fixed_parameters = {
        'dataset_filename': 'trec2007-1607201347',
        'label_type': {
            'ham_label': -1,
            'spam_label': 1,
        },
        'classifier': {
            'type': None,
            'training_parameters': {},
            'testing_parameters': {},
        },
        'attack': {
            'type': None,
            'parameters': {},
        },
    }

    ## put iteration last, but other dimensions is preference only
    experiment_batches = [
        [
            ('dataset',
                'dataset_filename',
                ['trec2007-1607201347', 'trec2007-1607252257', 'trec2007-1607252259']),
            ('classifier',
                ('classifier', 'type'),
                ['adaline', 'logistic regression']),
            ('attack',
                ('attack', 'type'),
                ['dictionary', 'focussed', 'empty', 'none']),
            ('% poisoned',
                ('attack', 'parameters', 'percentage_samples_poisoned'),
                [.0, .1, .2, .5]),
            ('learning rate',
                ('classifier', 'training_parameters', 'learning_rate'),
                [.005, .01, .02, .05, .1]),
            ('repetition',
                'repetition',
                range(1, 20+1)),
        ],
        [
            ('dataset',
                'dataset_filename',
                ['trec2007-1607201347', 'trec2007-1607252257', 'trec2007-1607252259']),
            ('classifier',
                ('classifier', 'type'),
                ['adaline', 'logistic regression', 'naive bayes']),
            ('attack',
                ('attack', 'type'),
                ['dictionary', 'focussed', 'empty', 'none']),
            ('% poisoned',
                ('attack', 'parameters', 'percentage_samples_poisoned'),
                [.0, .1, .2, .5]),
            ('repetition',
                'repetition',
                range(1, 20+1)),
        ]
    ]

    for parameter_ranges in experiment_batches:
        df = perform_experiment_batch(parameter_ranges, fixed_parameters, infolder)
        save_df(df, outfolder)

    return


if __name__ == '__main__':

    infolder = sys.argv[1] if len(sys.argv) > 1 else '../datasets/processed'
    outfolder = sys.argv[2] if len(sys.argv) > 2 else '.'
    sys.exit(main(infolder, outfolder))

