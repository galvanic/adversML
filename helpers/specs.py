#!/usr/bin/env python3
# coding: utf-8
from __future__ import division

'''
'''
from itertools import product
from collections import defaultdict


def generate_specs(experiment_dimensions):
    '''
    - specifications: details for how to carry out experiments, what
        parameters to use etc.

    TODO what parts of the experiment specs are tied together ? and can
         therefore be simplified ?
    '''
    static_params = {
        'dataset': 'trec2007',
        'dataset_filename': 'trec2007-1607201347',
        'label_type': {
            'ham_label': -1,
            'spam_label': 1,
        },
        'training_parameters': {},
        'testing_parameters': {},
    }

    dimensions, variations = zip(*experiment_dimensions.items())
    specs = [dict(zip(dimensions, values)) for values in product(*variations)]

    ## TODO refactor
    def add_recursive_key(key, dic, value):
        if type(key) == str:
            dic[key] = value
        elif len(key) == 1:
            add_recursive_key(key[0], dic, value)
        else:
            add_recursive_key(key[1:], dic[key[0]], value)

    ## nested specs
    specifications = []
    for spec in specs:
        specification = defaultdict(dict, **static_params)
        for key, value in spec.items():
            add_recursive_key(key, specification, value)
        specifications.append(dict(specification))

    return specifications

