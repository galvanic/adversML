# coding: utf-8
from __future__ import division

'''
'''
from itertools import product
from collections import defaultdict
from copy import deepcopy


def add_nested_key(dic, key, value):
    if type(key) == str:
        dic[key] = value
    elif len(key) == 1:
        add_nested_key(dic, key[0], value)
    else:
        add_nested_key(dic[key[0]], key[1:], value)


def generate_specs(parameter_ranges, fixed_parameters):
    '''
    Returns list of specifications, all the possible (the cartesian product)
    variations of experiment parameters, given the parameter ranges to vary

    Inputs:
    - parameter_ranges: parameters along which the experiments vary (called
      `dimensions` here) (eg. we vary the percentage of poisoned samples)
      A list of 2-tuples (keys, values):
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

    batch_id = fixed_parameters['batch_id']

    dimensions, ranges = zip(*parameter_ranges)
    specs = (dict(zip(dimensions, values)) for values in product(*ranges))

    ## turn tuple keys into nested dictionaries
    specifications = []
    for ii, spec in enumerate(specs):

        specification = defaultdict(dict)
        specification.update({'experiment_id': '%s %s' % (batch_id, ii)})
        specification.update(fixed_parameters)

        for key, value in spec.items():
            add_nested_key(specification, key, value)

        specifications.append(deepcopy(dict(specification)))

    return specifications

