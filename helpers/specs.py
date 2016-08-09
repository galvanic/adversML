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

    batch_id = fixed_parameters['experiment_batch_id']

    dimensions, ranges = zip(*parameter_ranges)
    specs = (dict(zip(dimensions, values)) for values in product(*ranges))

    ## turn tuple keys into nested dictionaries
    specifications = []
    for ii, spec in enumerate(specs):

        specification = defaultdict(dict)
        specification.update({'experiment_id': '%s_%s' % (batch_id, ii)})
        specification.update(fixed_parameters)

        for key, value in spec.items():
            add_nested_key(specification, key, value)

        specifications.append(deepcopy(dict(specification)))

    return specifications


from classifiers import adaline as AdalineClassifier
from classifiers import naivebayes as NaivebayesClassifier
from classifiers import logistic_regression as LogisticRegressionClassifier

from attacks import empty as EmptyAttack
from attacks import ham as HamAttack
from attacks import dictionary as DictionaryAttack
from attacks import focussed as FocussedAttack

class no_attack():
    def apply(features, labels, **kwargs):
        return features, labels

Classifiers = {
    'adaline':  AdalineClassifier,
    'naivebayes': NaivebayesClassifier,
    'logisticregression': LogisticRegressionClassifier,
}

Attacks = {
    'empty': EmptyAttack,
    'ham': HamAttack,
    'dictionary': DictionaryAttack,
    'focussed': FocussedAttack,
    'none': no_attack,
}


def prepare_spec(spec):
    '''
    Returns the experiment dictionary specification ready to carry out the
    experiment.
    For example, it duplicates certain keys so that the user doesn't have to
    enter them more than once (would increase chance of errors) and replaces
    None by actual objects (like a function that does nothing for the empty
    attack but would have been faff for user to write).

    TODO raise exceptions if doesn't exist, or catch KeyError
    '''
    spec = deepcopy(spec)

    ham_label = spec['label_type']['ham_label']
    spec['classifier']['training_parameters']['ham_label'] = ham_label
    spec['classifier']['testing_parameters' ]['ham_label'] = ham_label

    normalise_key = lambda k: k.lower().replace(' ', '')

    ## classifier
    classifier = spec['classifier']['type']
    classifier = Classifiers[normalise_key(classifier)]
    spec['add_bias'] = True if classifier != NaivebayesClassifier else False
    spec['classifier']['type'] = classifier

    ## attack
    attack = spec['attack']['type']
    attack = 'none' if not attack else attack
    attack = Attacks[normalise_key(attack)]
    spec['attack']['type'] = attack

    return spec

