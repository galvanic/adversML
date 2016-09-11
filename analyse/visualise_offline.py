#!/usr/bin/env python3
# coding: utf-8

import sys
import os.path
import glob

import yaml

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def visualise(df, dimension_rows, dimension_color, xaxis_label=None, custom_params={}):
    ''''''
    df = df.copy()

    params = {
        'ylim': (0, .5),
        'legend': None,
        'marker': '.',
        'markersize': 10,
    }
    params = {**params, **custom_params}

    xticklabels = list(np.arange(0, 60, 10))
    xticklabels.pop(0)
    xticklabels.insert(0, '')

    ## decide on dimension to visualise over the rows
    ## TODO reorder df levels to put that chosen dimension in front

    metrics = df.columns

    if not dimension_rows:
        df['none'] = 0
        levels_names = list(df.index.names)
        levels_names.insert(0, 'none')
        df = df.set_index('none', append=True)
        df = df.reorder_levels(levels_names)
        dimension_rows = 'none'

    rows = [lvl for lvl in df.index.levels if lvl.name == dimension_rows][0]
    rows = list(set(df.index.get_level_values(dimension_rows)))

    if dimension_color:
        df = df.unstack([dimension_color])

    xlabel = xaxis_label or df.index.names[-1]

    ## start plotting
    fig = plt.figure(figsize=(16, 4*len(rows)))

    ## AUC curves
    axes_auc = gridspec.GridSpec(len(rows), 1)
    axes_auc.update(left=0, right=0.19, wspace=0.1, hspace=0.1)

    for idx, row in enumerate(rows):
        axis = plt.subplot(axes_auc[idx, 0])

        title = 'AUC' if idx == 0 else None
        df.ix[row]['AUC'].plot(ax=axis, title=title, **params)

        axis.set_ylim(.5, 1)
        axis.set_ylabel(row)
        axis.set_xticklabels([])
        axis.set_xlabel('')

        if idx == len(rows)-1:
            axis.set_xticklabels(xticklabels)

    ## other metrics
    axes = gridspec.GridSpec(len(rows), len(metrics)-1)
    axes.update(left=0.25, right=1, wspace=0.1, hspace=0.1)

    for idx, row in enumerate(rows):
        for ic, metric in enumerate(metrics[1:]):
            axis = plt.subplot(axes[idx, ic])

            title = metric if idx == 0 else None
            df.ix[row][metric].plot(ax=axis, title=title, sharey=True, **params)

            axis.set_xticklabels([])
            axis.set_xlabel('')

            if idx == len(rows)-1:
                if ic == 0:
                    axis.set_xticklabels(xticklabels)
                if ic == len(metrics)-2:
                    axis.set_xlabel(xlabel)

            ## legend
            elif idx == 0 and ic == 3:
                axis.legend()
                _, labels = axis.get_legend_handles_labels()
                new_labels = map(lambda l: l.replace('(', '').replace(')', ''), labels)
                axis.legend(frameon=False, ncol=2, borderaxespad=0, numpoints=1, markerfirst=True, markerscale=0,
                            loc='upper center', bbox_to_anchor=(.83, .99), bbox_transform=fig.transFigure,
                            labels=list(new_labels))

    plt.show()
    return


### HELPERS

def get_batch_df(batch_id, dataset_dir):
    ifilepath = os.path.join(dataset_dir, '%s-df.dat' % batch_id)
    df = pd.read_pickle(ifilepath)
    return df


def main(batch_id, dataset_dir, group_size):
    '''
    '''
    df = get_batch_df(batch_id, dataset_dir)
    experiment_ids = get_sorted_experiment_ids(df)

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

    sys.exit(main(batch_id, dataset_dir, group_size))

