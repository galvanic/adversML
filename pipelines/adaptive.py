# coding: utf-8
from __future__ import division
'''
'''
import os
import pickle
import numpy as np
import pandas as pd
from pprint import pformat

from helpers.logging import tls, log
from helpers.performance import get_error, get_FPR, get_FNR, get_ROC_AUC
from helpers.gradientdescent import get_cost

sigmoid = lambda z: 1 / (1 + np.exp(-z))
tanh = lambda z: np.tanh(z)

## helpers
add_bias = lambda x: np.insert(x, 0, values=1, axis=1) # add bias term

def get_prediction(df, classifier_name):
    column = np.sign(df[(classifier_name, 'output')])
    column = column.rename((classifier_name, 'prediction'))
    try:
        column = column.astype(np.int8)
    except ValueError:
        pass
    return column

def get_loss(df, classifier_name):
    column = (df[(classifier_name, 'prediction')] - df['y']) ** 2
    column = column.rename((classifier_name, 'loss'))
    try:
        column = column.astype(np.int8)
    except ValueError:
        pass
    return column


@log(get_experiment_id=lambda args: args[0]['experiment_id'])
def run_experiment(spec):
    '''
    Returns the performance of the experiment defined by the given specification

    Inputs:
    - spec: specifications of the experiment
        A dictionary

    Outputs:
    - performance: metrics of the performance of the trained classifier
        under the specified attack
        A dictionary
    '''

    tls.logger.info('spec:\n%s\n' % pformat(spec))
    spec = prepare_spec(spec)
    tls.logger.info('prepared spec:\n%s\n' % pformat(spec))

    ifilepath = os.path.join(spec['dataset_dirpath'], spec['dataset_filename'])
    with open('%s-features.dat' % ifilepath, 'rb') as infile:
        X = pickle.load(infile)

    with open('%s-labels.dat' % ifilepath, 'rb') as infile:
        Y = pickle.load(infile)

    ## normalise Y labels to (-1, 1)
    if tuple(np.unique(Y)) == (0, 1):
        Y = np.array(Y, dtype=np.int8) * 2 - 1

    N, D = X.shape
    tls.logger.debug('X: (%s, %s)\tY: %s' % (N, D, str(Y.shape)))

    ## split dataset into training and testing sets
    permuted_indices = np.random.permutation(N)
    X = X[permuted_indices]
    Y = Y[permuted_indices]

    N_test = 1000

    X_train = X[:-N_test]
    Y_train = Y[:-N_test]
    X_test  = X[-N_test:]
    Y_test  = Y[-N_test:]

    ## apply attack:
    for attack_spec in spec['attack']:
        attack = attack_spec['type']
        attack_params = attack_spec['parameters']
        if attack_params['percentage_samples_poisoned'] != 0:
            X_train, Y_train = attack.apply(features=X_train, labels=Y_train, **attack_params)

    ## prepare dataset
    if spec['add_bias']:
        X_train, X_test = map(add_bias, [X_train, X_test])
        tls.logger.debug('- added bias')

    ## run simulation
    classifier_fast = spec['classifier_fast']
    classifier_slow = spec['classifier_slow']

    results = run(X_train, Y_train, X_test, Y_test,
                  classifier_fast, classifier_slow,
                  **spec['training_parameters'])

    ## collect performance measures
    df = pd.DataFrame.from_dict(results)
    df.index = df.index.set_names(['timestep'])
    df.columns = df.columns.set_names(['classifier', 'metrics'])
    df = df.join(pd.Series(np.ravel(Y_train)).rename(('y', '')))

    classifier_names = ['fast', 'slow', 'combination']
    for name in classifier_names:
        df = df.join(get_prediction(df, name))
        df = df.join(get_loss(df, name))

    df = df.sort_index(axis=1)
    df = df.ix[1:]

    tls.logger.info('performance:\n%s' % df.ix[len(df)-4:].to_string(col_space=8, float_format=lambda x: '%.3f' % x))
    df_row = df.unstack('timestep').to_frame().T

    ## release memory
    del X
    del Y
    del X_train
    del Y_train
    del X_test
    del Y_test

    return df_row

from collections import defaultdict
from collections import deque

WINDOW_OPERATORS = {
    'mean': np.mean,
    'median': np.median,
}

def get_error_gradient(windows, y, λ, operator=np.mean):
    ''''''
    o, o_1, o_2 = map(np.array, windows)
    e = (y - o)
    error_gradient = e * (o_1 - o_2) * λ * (1-λ)
    return operator(error_gradient, axis=0)


@log
def run(X, Y, X_test, Y_test,
        ## params
        classifier1,
        classifier2,
        adaptation_rate,
        window_size,
        window_operator,
        ):
    '''
    '''
    ## keep track of various metrics
    ## tuple keys will become DataFrame MultiIndex
    record = defaultdict(list)
    '''
    record.update({
        'fast': defaultdict(list),
        'slow': defaultdict(list),
        'combination': defaultdict(list),
    })
    '''

    ## setup
    N, D = X.shape           # N #training samples; D #features
    tls.logger.debug('X: (%s, %s)\tY: %s' % (N, D, str(Y.shape)))
    a = 0.5              ## λ is modified indirectly via a (see paper)
    λ = sigmoid(a)       ## mixing parameter
    η = adaptation_rate  ## learning parameter for the mixing parameter (adaptation speed)
    η1 = classifier1['training_parameters']['learning_rate']
    η2 = classifier2['training_parameters']['learning_rate']
    classifier1 = classifier1['type']
    classifier2 = classifier2['type']
    operator = WINDOW_OPERATORS[window_operator]
    window_o  = deque(maxlen=window_size)
    window_o1 = deque(maxlen=window_size)
    window_o2 = deque(maxlen=window_size)

    ## initialise weights for both filters
    W_1 = np.zeros((D, 1))
    W_2 = np.zeros((D, 1))

    for sample in range(N):
        if sample % 1000 == 0: tls.logger.info('iteration %d' % sample)
        x, y = X[sample], Y[sample]

        ###
        ### update each part of the model
        ###

        ## update the weights for each classifier
        o_1 = classifier1.compute_output(x, W_1)
        gradient = np.multiply((o_1 - y), x)
        W_1 = W_1 - η1 * gradient.reshape(W_1.shape)

        o_2 = classifier2.compute_output(x, W_2)
        gradient = np.multiply((o_2 - y), x)
        W_2 = W_2 - η2 * gradient.reshape(W_2.shape)

        ## update mixing parameter λ via a's update equation (5) from the paper
        o = λ * o_1 + (1-λ) * o_2  ## combining outputs
        window_o.append(o)
        window_o1.append(o_1)
        window_o2.append(o_2)
        error_gradient = get_error_gradient([window_o, window_o1, window_o2], y, λ, operator)
        a_temp = a + η * error_gradient

        a = a_temp[0] if sigmoid(a_temp) < 0.85 and sigmoid(a_temp) > 0.15 else a
        ## note: extract unique value [0] because of array shape

        λ = sigmoid(a)
        #tls.logger.info('  λ: %.2f' % λ)
        record[('λ', '')].append(λ)

        ###
        ### keep track of output and predictions on sample for all classifiers
        ###

        record[('fast', 'output')].append(o_1[0])
        record[('slow', 'output')].append(o_2[0])
        record[('combination', 'output')].append(o[0])

        ## threshold, cost, error can be calculated from this
        ## using the true label, available outside the function
        ## and all these can be used to compute the cumulative loss
        ## over a time window, to calculate the regret

        ###
        ### measure performance on test set
        ###

        ## idea is to see how these weights compare to a whole "test" set
        ## even though point is to use them on a non-stationary distribution
        ## so not that useful

        ## compute performance of both individual classifiers on the (entire) test set
        O_1 = classifier1.compute_output(X_test, W_1)
        T_1 = classifier1.compute_prediction(O_1)
        cost1 = get_cost(Y_test, O_1)
        error1 = get_error(Y_test, T_1)
        #tls.logger.debug('  classifier 1 cost: %.3f' % cost1)
        #tls.logger.debug('  classifier 1 error: %.3f' % error1)
        record[('fast', 'cost')].append(cost1)
        record[('fast', 'error')].append(error1)

        O_2 = classifier2.compute_output(X_test, W_2)
        T_2 = classifier2.compute_prediction(O_2)
        cost2 = get_cost(Y_test, O_2)
        error2 = get_error(Y_test, T_2)
        #tls.logger.debug('  classifier 2 cost: %.3f' % cost2)
        #tls.logger.debug('  classifier 2 error: %.3f' % error2)
        record[('slow', 'cost')].append(cost2)
        record[('slow', 'error')].append(error2)

        O = λ * O_1 + (1-λ) * O_2  ## combining outputs
        ## compute performance of the adaptive combination on the test set
        T = np.sign(O)
        cost = get_cost(Y_test, O)
        error = get_error(Y_test, T)
        #tls.logger.debug('  combination cost: %.3f' % cost)
        #tls.logger.debug('  combination error: %.3f' % error)
        record[('combination', 'cost')].append(cost)
        record[('combination', 'error')].append(error)

    return record


from copy import deepcopy
from itertools import chain

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

normalise_key = lambda k: k.lower().replace(' ', '')


def get_attack_dict_with_func(attack_key):

    if type(attack_key) == dict:
        attack = deepcopy(attack_key)
        attack_type = attack['type']
        attack_type = 'none' if not attack_type else attack_type
        attack_type = Attacks[normalise_key(attack_type)]
        attack['type'] = attack_type
        return [attack]

    elif type(attack_key) == list:
        return list(chain(*map(get_attack_dict_with_func, attack_key)))

    return


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

    for classifier_type in ['fast', 'slow']:

        classifier_key = 'classifier_%s' % classifier_type

        classifier = spec[classifier_key]
        classifier['training_parameters']['ham_label'] = ham_label
        classifier['testing_parameters' ]['ham_label'] = ham_label

        ## classifier
        classifier = classifier['type']
        classifier = Classifiers[normalise_key(classifier)]
        spec['add_bias'] = True if classifier != NaivebayesClassifier else False
        spec[classifier_key]['type'] = classifier

    ## attacks
    spec['attack'] = get_attack_dict_with_func(spec['attack'])

    return spec

