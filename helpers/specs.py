#!/usr/bin/env python3
# coding: utf-8
from __future__ import division

'''
'''
from itertools import product
from collections import defaultdict


def generate_specs(parameter_ranges, fixed_parameters):
    '''
    Returns list of specifications, all the possible (the cartesian product)
    variations of experiment parameters, given the parameter ranges to vary

    Inputs:
    - parameter_ranges: parameters along which the experiments vary (called
      `dimensions` here) (eg. we vary the percentage of poisoned samples)
      A dictionary or OrderedDict with:
      - keys: string or tuple, with tuples as keys for nested dictionaries to
              represent the final spec key
              eg.: ('attack_params', 'percentage_samples_poisoned') corresponds
              to {'attack_params': {'percentage_samples_poisoned'}}
      - values: Iterable, corresponding to range of values that parameter is
              being tested at
    - fixed_parameters: specs that don't change
      A dictionary

    Outputs:
    - specifications: details for how to carry out experiments, what parameters
      to use etc.
      list of dictionaries

    TODO what parts of the experiment specs are tied together ? and can
         therefore be simplified ?
    '''
    dimensions, ranges = zip(*parameter_ranges.items())
    specs = [dict(zip(dimensions, values)) for values in product(*ranges)]

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
        specification = defaultdict(dict, **fixed_parameters)
        for key, value in spec.items():
            add_recursive_key(key, specification, value)
        specifications.append(dict(specification))

    return specifications

