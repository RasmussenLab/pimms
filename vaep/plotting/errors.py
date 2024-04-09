"""Plot errors based on DataFrame with model predictions."""
from __future__ import annotations

import itertools
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from seaborn.categorical import _BarPlotter

import vaep.pandas.calc_errors


def plot_errors_binned(pred: pd.DataFrame, target_col='observed',
                       ax: Axes = None,
                       palette: dict = None,
                       metric_name: Optional[str] = None,
                       errwidth: float = 1.2) -> Axes:
    assert target_col in pred.columns, f'Specify `target_col` parameter, `pred` do no contain: {target_col}'
    models_order = pred.columns.to_list()
    models_order.remove(target_col)
    errors_binned = vaep.pandas.calc_errors.calc_errors_per_bin(
        pred=pred, target_col=target_col)

    meta_cols = ['bin', 'n_obs']  # calculated along binned error
    len_max_bin = len(str(int(errors_binned['bin'].max())))
    n_obs = (errors_binned[meta_cols]
             .apply(
        lambda x: f"{x.bin:0{len_max_bin}}\n(N={x.n_obs:,d})", axis=1
    )
        .rename('intensity bin')
        .astype('category')
    )
    metric_name = metric_name or 'Average error'

    errors_binned = (errors_binned
                     [models_order]
                     .stack()
                     .to_frame(metric_name)
                     .join(n_obs)
                     .reset_index()
                     )

    ax = sns.barplot(data=errors_binned, ax=ax,
                     x='intensity bin', y=metric_name, hue='model',
                     palette=palette,
                     errwidth=errwidth,)
    ax.xaxis.set_tick_params(rotation=90)
    return ax, errors_binned


def plot_errors_by_median(pred: pd.DataFrame,
                          feat_medians: pd.Series,
                          target_col='observed',
                          ax: Axes = None,
                          palette: dict = None,
                          feat_name: str = None,
                          metric_name: Optional[str] = None,
                          errwidth: float = 1.2) -> tuple[Axes, pd.DataFrame]:
    # calculate absolute errors
    errors = vaep.pandas.get_absolute_error(pred, y_true=target_col)
    errors.columns.name = 'model'

    # define bins by integer value of median feature intensity
    feat_medians = feat_medians.astype(int).rename("bin")

    # number of intensities per bin
    n_obs = pred[target_col].to_frame().join(feat_medians)
    n_obs = n_obs.groupby('bin').size().to_frame('n_obs')

    errors = (errors
              .stack()
              .to_frame(metric_name)
              .join(feat_medians)
              ).reset_index()
    n_obs.index.name = "bin"

    errors = errors.join(n_obs, on="bin")

    if feat_name is None:
        feat_name = feat_medians.index.name
        if not feat_name:
            feat_name = 'feature'

    x_axis_name = f'intensity binned by median of {feat_name}'
    len_max_bin = len(str(int(errors['bin'].max())))
    errors[x_axis_name] = (
        errors[['bin', 'n_obs']]
        .apply(
            lambda x: f"{x.bin:0{len_max_bin}}\n(N={x.n_obs:,d})", axis=1
        )
        .rename('intensity bin')
        .astype('category')
    )

    metric_name = metric_name or 'Average error'

    sns.barplot(data=errors,
                ax=ax,
                x=x_axis_name,
                y=metric_name,
                hue='model',
                palette=palette,
                errwidth=errwidth,)
    ax.xaxis.set_tick_params(rotation=90)
    return ax, errors


def get_data_for_errors_by_median(errors: pd.DataFrame, feat_name, metric_name):
    """Extract Bars with confidence intervals from seaborn plot.
      Confident intervals are calculated with bootstrapping (sampling the mean).

    Relies on internal seaborn class. only used for reporting of source data in the paper.
    """
    x_axis_name = f'intensity binned by median of {feat_name}'

    plotter = _BarPlotter(data=errors, x=x_axis_name, y=metric_name, hue='model',
                          order=None, hue_order=None,
                          estimator="mean", errorbar=("ci", 95), n_boot=1000, units=None, seed=None,
                          orient=None, color=None, palette=None, saturation=.75, width=.8,
                          errcolor=".26", errwidth=None, capsize=None, dodge=True)
    ax = plt.gca()
    plotter.plot(ax, {})
    plt.close(ax.get_figure())
    mean, cf_interval = plotter.statistic.flatten(), plotter.confint.reshape(-1, 2)
    plotted = pd.DataFrame(np.concatenate((mean.reshape(-1, 1), cf_interval), axis=1), columns=[
        'mean', 'ci_low', 'ci_high'])
    _index = pd.DataFrame(list(itertools.product(
        (_l.get_text() for _l in ax.get_xticklabels()),  # bins x-axis
        (_l.get_text() for _l in ax.get_legend().get_texts()),  # models legend
    )
    ), columns=['bin', 'model'])
    plotted = pd.concat([_index, plotted], axis=1)
    return plotted


def plot_rolling_error(errors: pd.DataFrame, metric_name: str, window: int = 200,
                       min_freq=None, freq_col: str = 'freq', colors_to_use=None,
                       ax=None):
    errors_smoothed = errors.drop(freq_col, axis=1).rolling(
        window=window, min_periods=1).mean()
    errors_smoothed_max = errors_smoothed.max().max()
    errors_smoothed[freq_col] = errors[freq_col]
    if min_freq is None:
        min_freq = errors_smoothed[freq_col].min()
    else:
        errors_smoothed = errors_smoothed.loc[errors_smoothed[freq_col] > min_freq]
    ax = errors_smoothed.plot(x=freq_col, ylabel=f'rolling average error ({metric_name})',
                              color=colors_to_use,
                              xlim=(min_freq, errors_smoothed[freq_col].max()),
                              ylim=(0, min(errors_smoothed_max, 5)),
                              ax=None)
    return ax
