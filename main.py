#!/usr/bin/env python3
# coding: utf-8
from __future__ import division

'''
'''
import sys

from pipeline import perform_experiment_batch


def main():

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

    return df


if __name__ == '__main__':
    sys.exit(main())

