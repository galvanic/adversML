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


def get_attacks_info(attack_key):
    '''Return list of dictionaries with information for visualising the attack'''

    if type(attack_key) == dict:
        info = attack_key['parameters']
        if 'duration' not in info:
            info.update({'duration': info['end'] - info['start']})
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


def get_legend_labels(df, latexify=True):
    try:
        labels = df.columns.get_level_values('classifier')
    except AttributeError:
        labels = [df.name]

    if latexify:
        labels = map(lambda l: '$%s$' % l, labels)
    return labels


def visualise(df, axes, idx=0, title='', attacks=None, **custom_params):
    '''visualise 1 experiment'''

    PARAMS = {
        'sharex': True,
        'ylim': (0, 1),
        'legend': None,
    }

    columns = [df.xs('cost',  axis=1, level='metrics'),
               df.xs('error', axis=1, level='metrics'),
               df['λ'],]
    labels = map(get_legend_labels, columns)

    for ii, (xs, label) in enumerate(zip(columns, labels)):

        axis = plt.subplot(axes[ii, idx])
        params = {**deepcopy(PARAMS), **custom_params}

        ## plot attack
        edgecolor = 'none' if len(attacks) == 1 else 'black'
        if attacks:
            for attack in attacks:
                axis.add_patch(patches.Rectangle(
                    (attack['start'], 0), attack['duration'], 1,
                     facecolor='gray', edgecolor=edgecolor, alpha=0.1))

        ## metric is λ
        if ii == 2:
            params.update({'color': 'black'})

        xs.plot(x=df.index, y=['fast', 'slow', 'combination'], ax=axis, **params)

        ## left-most graph shows Y axis labels, others hide them
        axis.set_yticklabels([])
        if idx == 0:
            axis.set_yticklabels([''] + list(np.arange(0.2, 1.2, 0.2)),
                                 position=(-0.01, 0), verticalalignment='top')

        ## right-most graph hosts legend
        if idx == axes._ncols-1:
            axis.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5),
                        labels=list(label), fontsize=14)

            ## metric is λ
            if ii == 2:
                plt.yticks([0.15, 0.85], ['slow', 'fast'])
                axis.yaxis.set_ticks_position('right')
                for tick, color in zip(axis.yaxis.get_ticklabels(), ['green', 'blue']):
                    tick.set_color(color)

        ## top-most graph hosts title
        if ii == 0:
            plt.title(title, fontsize=16, y=1.1)

    ## don't show every other X axis label, too crowded
    for label in axis.xaxis.get_ticklabels()[1::2]: label.set_visible(False)

    return


def visualise_group(dfs, figsize=(27, 7), **custom_params):
    '''visualise multiple experiments'''
    fig = plt.figure(figsize=figsize)
    axes = gridspec.GridSpec(3, len(dfs), height_ratios=[2, 2, 1])
    axes.update(hspace=0, wspace=0.05)

    for idx, (df, title, attacks) in enumerate(dfs):

        visualise(prepare(df), axes, idx, title, attacks, **custom_params)

    plt.show()
    return


def compare_group(df, experiment_ids_group, attacks_info, **custom_params):
    '''Plot the group horizontally, one graph per experiment in the group'''

    to_plot = []
    for experiment_id in experiment_ids_group:
        xs = get_xs(experiment_id, df)
        attacks = get_specific_attack_info(xs, attacks_info)

        experiment_num = experiment_id.split(' ')[1]
        title = 'experiment %s\n%s' % (experiment_num, get_title(xs))

        to_plot.append((xs, title, attacks))

    visualise_group(to_plot, **custom_params)
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


def get_attacks_info_from_specs(batch_specs):
    attacks = get_attacks_info(batch_specs['default_parameters']['attack'])
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
        compare_group(df, experiment_ids_group, attacks, **custom_params)

        from code import interact; interact(local=dict(locals(), **globals()))

    return


if __name__ == '__main__':

    dataset_dir, batch_id = os.path.split(sys.argv[1])
    batch_id = batch_id.split('.')[0]

    group_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    sys.exit(main(batch_id, dataset_dir, group_size))

