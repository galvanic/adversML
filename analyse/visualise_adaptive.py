#!/usr/bin/env python3
# coding: utf-8

import sys
import os.path
import glob

import yaml
from pprint import pprint
from itertools import chain
from copy import deepcopy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
pd.set_option('display.precision', 4)


###
### HELPERS
###


def get_attacks_info(attack_key, N):
    '''Return list of dictionaries with information for visualising the attack'''

    if type(attack_key) == dict:
        info = attack_key['parameters']
        info['start'] = int(info['start'] * N)
        info['duration'] = int(info['duration'] * N)
        return [info]

    elif type(attack_key) == list:
        return list(chain(*map(get_attacks_info, attack_key)))


def chunks(lst, groups_of):
    '''Yield successive n-sized chunks from list'''
    for i in range(0, len(lst), groups_of):
        yield lst[i:i + groups_of]


def get_xs(experiment_id, df):
    '''Return cross-section of df for given experiment_id'''
    return df.xs(experiment_id, level='experiment_id')


def shorten(name):
    '''Replace certain recognised names with shorter alternatives'''
    if type(name) == str:
        name = 'LogReg' if name == 'logistic regression' else name.capitalize()
    return name


def latexify(name):
    '''Format text that can be, into latex word with subscript'''
    try:
        name = '%s_{%s}' % tuple(name.split(' '))
    except TypeError:
        pass
    return name


def get_title(df):
    '''Return latex'ed title for the graph, with each attribute that varies'''
    idx = df.index

    try:
        levels = idx.levels
        labels = idx.labels
    except AttributeError:
        levels = [idx]
        labels = [[0]]

    params = []
    for name, level, label in zip(idx.names, levels, labels):

        key = name
        value = level[label][0]
        params.append('$%s: %s$' % (latexify(key), shorten(value)))

    return '\n'.join(params)


def get_specific_attack_info(df, attacks):
    '''Return dictionary with updated information for attack in case this
    information varies and therefore wasn't in the default parameters
    Assumes only one attack and not a list'''
    attack = deepcopy(attacks)
    if 'attack start' in df.index.names:
        attack.update({'start': df.index.get_level_values('attack start')[0]})
    if 'attack duration' in df.index.names:
        attack.update({'duration': df.index.get_level_values('attack duration')[0]})
    return attack


def prepare(df):
    '''Format the dataframe to be passed to visualisation function'''
    return df.stack('timestep').reset_index(df.index.names, drop=True)


###
### Plotting functions
###


def visualise(df, plot_specs, axes, idx=0, title='', attacks=[], **custom_params):
    '''visualise 1 experiment'''

    PARAMS = {
        'sharex': True,
        'legend': None,
    }

    patch_params = {
        'facecolor': 'gray',
        'edgecolor': 'none' if len(attacks) == 1 else 'black',
        'alpha': 0.1
    }

    for ii, spec in enumerate(plot_specs):

        axis = plt.subplot(axes[ii, idx])
        params = {**deepcopy(PARAMS), **custom_params, **spec['specific_params']}

        xs = spec['get_data'](df)

        for y, color in spec['y_colors'].items():
            xs.plot(y=y, ax=axis, color=color, **params)

        ## draw attack
        for attack in attacks:
            axis.add_patch(patches.Rectangle(
                xy = (attack['start'], axis.get_ylim()[0]),
                width = attack['duration'],
                height = axis.get_ylim()[1] - axis.get_ylim()[0],
                **patch_params))

        ## left-most graph shows Y axis labels, others hide them
        if idx != 0:
            axis.set_yticklabels([])
        else:
            #from code import interact; interact(local=dict(locals(), **globals()))
            y_labels = spec['y_ticklabels'] if 'y_ticklabels' in spec else plt.yticks()[0][1:]
            axis.set_yticklabels([''] + list(y_labels),
                                 position=(-0.001, 0), verticalalignment='top')

        ## right-most graph hosts legend
        if idx == axes._ncols-1:
            axis.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5),
                        labels=list(spec['y_labels']), fontsize=14)

            ## metric is λ
            ## TODO put this in outside function so that it can only be run for λ ?
            if ii == len(plot_specs)-1:
                y_labels = ['slow', 'fast']
                plt.yticks([0.15, 0.85], y_labels)
                axis.tick_params(axis='y', which='both', labelleft='off', labelright='on')
                axis.set_yticklabels(y_labels, position=(1, 0), verticalalignment='center')
                for tick, color in zip(axis.yaxis.get_ticklabels(), ['green', 'blue']):
                    tick.set_color(color)

        ## top-most graph hosts title
        if ii == 0:
            plt.title(title, fontsize=16, y=1.1)

    ## don't show every other X axis label, too crowded
    for label in axis.xaxis.get_ticklabels()[1::2]: label.set_visible(False)

    return


def visualise_group(dfs, plot_specs, figsize=(27, 7), **custom_params):
    '''visualise multiple experiments
    dfs is horizontally
    plot_specs is vertically (locally)'''
    fig = plt.figure(figsize=figsize)
    axes = gridspec.GridSpec(3, len(dfs), height_ratios=[2, 2, 1])
    axes.update(hspace=0, wspace=0.05)

    for idx, (df, title, attacks) in enumerate(dfs):

        visualise(df, plot_specs, axes, idx, title, attacks, **custom_params)

    plt.show()
    return


identity = lambda x: x

def compare_group(df, experiment_ids_group, plot_specs, attacks_info,
        transform=identity,
        **custom_params):
    '''Plot the group horizontally, one graph per experiment in the group'''

    if type(plot_specs) == str:
        plot_specs = get_existing_plot_specs(plot_specs)
        plot_specs = update_plot_specs(plot_specs)

    to_plot = []
    for experiment_id in experiment_ids_group:
        xs = get_xs(experiment_id, df)
        attacks = get_specific_attack_info(xs, attacks_info)

        experiment_num = experiment_id.split(' ')[1]
        title = 'experiment %s\n%s' % (experiment_num, get_title(xs))

        xs = prepare(xs)   ## unstack timestep
        xs = transform(xs)
        to_plot.append((xs, title, attacks))

    visualise_group(to_plot, plot_specs, **custom_params)
    return


###
### small helpers for main
###


def get_batch_specs(batch_id, dataset_dir):
    '''Returns dictionary of batch specifications'''
    try:
        specs_filepath = glob.glob(os.path.join(dataset_dir, '%s*.y*ml' % batch_id))[0]
    except IndexError:
        print('Could not find file')
        return

    with open(specs_filepath, 'r') as ifile:
        batch_specs = yaml.safe_load(ifile)
    return batch_specs


def get_attacks_info_from_specs(batch_specs, N):
    attacks = get_attacks_info(batch_specs['default_parameters']['attack'], N)
    return attacks


def get_batch_df(batch_id, dataset_dir):
    ifilepath = os.path.join(dataset_dir, '%s-df.dat' % batch_id)
    df = pd.read_pickle(ifilepath)
    return df


def get_sorted_experiment_ids(df):
    by_experiment_num = lambda label: int(label.split(' ')[1])
    experiment_ids = df.index.get_level_values('experiment_id')
    experiment_ids = sorted(experiment_ids, key=by_experiment_num)
    return experiment_ids


from collections import OrderedDict

CLASSIFIER_COLORS = OrderedDict([
    ('fast', 'blue'),
    ('slow', 'green'),
    ('combination', 'red'),
])

PLOT_SPECS = {
    'test set': [
        {
            'name': 'cost',
            'get_data': lambda df: df.xs('cost',  axis=1, level='metrics'),
            'y_colors': CLASSIFIER_COLORS,
            'specific_params': { 'ylim': (0, 1) },
        },
        {
            'name': 'error',
            'get_data': lambda df: df.xs('error', axis=1, level='metrics'),
            'y_colors': CLASSIFIER_COLORS,
            'specific_params': { 'ylim': (0, 1) },
        },
        {
            'name': 'λ',
            'get_data': lambda df: df,
            'y_colors': { 'λ': 'black' },
            'specific_params': { 'ylim': (0, 1) },
        },
    ],

    'regret': [
        {
            'name': 'regret',
            'get_data': lambda df: df,
            'y_colors': { 'regret': 'orange' },
        },
        {
            'name': 'cumulative loss',
            'get_data': lambda df: df.xs('cumulative loss', axis=1, level='metrics'),
            'y_colors': CLASSIFIER_COLORS,
        },
        {
            'name': 'λ',
            'get_data': lambda df: df,
            'y_colors': { 'λ': 'black' },
            'specific_params': { 'ylim': (0, 1) },
        },
    ],
}

## TODO ^ add possibility to change gridspec proportions for each etc.
## TODO ^ add outside options for whole graph, eg. transform function for df

def get_existing_plot_specs(name):
    ''''''
    return PLOT_SPECS[name]


def update_plot_specs(plot_specs):
    plot_specs = deepcopy(plot_specs)
    for spec in plot_specs:
        spec['specific_params'] = spec.get('specific_params', {})
        spec['y_labels'] = map(lambda l: '$%s$' % l, list(spec['y_colors'].keys()))
    return plot_specs


def main(batch_id, dataset_dir, group_size):
    '''
    '''

    specs = get_batch_specs(batch_id, dataset_dir)
    attacks = get_attacks_info_from_specs(specs)
    df = get_batch_df(batch_id, dataset_dir)
    experiment_ids = get_sorted_experiment_ids(df)

    total_timesteps = len(df.columns.get_level_values('timestep'))

    pprint(attacks)
    print('\ntotal timesteps: %d' % total_timesteps)


    for experiment_ids_group in chunks(experiment_ids, groups_of=group_size):

        custom_params = {}
        compare_group(df, experiment_ids_group,
            plot_specs='test set',
            attacks_info=attacks,
            **custom_params)

        from code import interact; interact(local=dict(locals(), **globals()))

    return


if __name__ == '__main__':

    dataset_dir, batch_id = os.path.split(sys.argv[1])
    batch_id = batch_id.split('.')[0]

    group_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    sys.exit(main(batch_id, dataset_dir, group_size))

