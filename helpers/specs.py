#!/usr/bin/env python3
# coding: utf-8
from __future__ import division

'''
'''
from itertools import product
from collections import defaultdict

from classifiers import adaline as AdalineClassifier
from classifiers import naivebayes as NaivebayesClassifier
from attacks import empty as EmptyAttack
from attacks import ham as HamAttack

class no_attack():
    def apply(features, labels, **kwargs):
        return features, labels

Classifiers = {
    'adaline':  AdalineClassifier,
    'naivebayes': NaivebayesClassifier,
}

Attacks = {
    'empty': EmptyAttack,
    'ham': HamAttack,
    'none': no_attack,
}


def process_experiment_declaration(experiment):
    '''
    Returns the experiment dictionary specification ready to carry out the
    experiment.
    For example, it duplicates certain keys so that the user doesn't have to
    enter them more than once (would increase chance of errors) and replaces
    None by actual objects (like a function that does nothing for the empty
    attack but would have been faff for user to write).

    TODO raise exceptions if doesn't exist, or catch KeyError
    '''
    ham_label = experiment['label_type']['ham_label']
    experiment['training_parameters']['ham_label'] = ham_label
    experiment['testing_parameters' ]['ham_label'] = ham_label

    normalise_key = lambda k: k.lower().replace(' ', '')

    experiment['classifier'] = Classifiers[normalise_key(experiment['classifier'])]
    experiment['add_bias'] = True if experiment['classifier'] != NaivebayesClassifier else False

    attack = Attacks[normalise_key(experiment['attack'])]
    attack = 'none' if not attack else attack
    experiment['attack'] = attack

    return experiment


def generate_specs():
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

    experimental_dimensions = {
        'classifier': ['adaline', 'naive bayes'],
        'attack': ['ham', 'empty'],
        ('attack_parameters', 'percentage_samples_poisoned'): [.1, .5],
    }

    dimensions, variations = zip(*experimental_dimensions.items())
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

    specifications = map(process_experiment_declaration, specifications)

    return specifications

