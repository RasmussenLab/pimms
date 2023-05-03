"""Plot data distribution based on pandas DataFrames or Series."""
from typing import Tuple, Iterable

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns


def min_max(s: pd.Series) -> Tuple[int]:
    min_bin, max_bin = (int(s.min()), (int(s.max())+1))
    return min_bin, max_bin


def get_min_max_iterable(series: Iterable[pd.Series]) -> Tuple[int]:
    """Get the min and max as integer from an iterable of pandas.Series."""
    min_bin = int(
        min(
            (s.min() for s in series))
    )
    max_bin = int(
        max(
            s.max() for s in series)
    )
    return min_bin, max_bin


def plot_histogram_intensites(s: pd.Series,
                              interval_bins=1,
                              min_max=(15, 40),
                              ax=None,
                              **kwargs) -> Tuple[Axes, range]:
    min_bin, max_bin = min_max
    bins = range(min_bin, max_bin, interval_bins)
    ax = s.plot.hist(bins=bins, xticks=list(bins),
                     ax=ax, **kwargs)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    return ax, bins


def plot_observations(df: pd.DataFrame,
                      ax: Axes = None,
                      title: str = '',
                      axis: int = 1,
                      ylabel: str = 'number of features',
                      xlabel: str = 'Samples ordered by number of features') -> Axes:
    """Plot non missing observations by row (axis=1) or column (axis=0) in
    order of number of available observations.
    No binning is applied, only counts of non-missing values are plotted.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame on which `notna` is applied
    ax : Axes, optional
        Axes to plot on, by default None
    title : str, optional
        Axes title, by default ''
    axis : int, optional
        dimension to sum over, by default 1
    ylabel : str, optional
        y-Axis label, by default 'number of features'
    xlabel : str, optional
        x-Axis label, by default 'Samples ordered by number of features'

    Returns
    -------
    Axes
        Axes on which plot was plotted
    """
    ax = (df
          .notna()
          .sum(axis=axis)
          .sort_values()
          .reset_index(drop=True)
          .plot(
              ax=ax,
              style='.',
              title=title,
              ylabel=ylabel,
              xlabel=xlabel)
          )
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    return ax


def plot_missing_dist_highdim(data: pd.DataFrame,
                              min_feat_per_sample=None,
                              min_samples_per_feat=None) -> matplotlib.figure.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    not_na = data.notna()
    name = 'features per sample'
    ax = (not_na
          .sum(axis=1)
          .to_frame(name)
          .groupby(name)
          .size()
          .sort_index()
          .plot
          .line(style='-',
                ax=axes[0])
          )
    ax.set_ylabel('observations (samples)')
    if min_feat_per_sample is not None:
        ax.vlines(min_feat_per_sample, *ax.get_ylim(), color='red')
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    name = 'samples per feature'
    ax = (not_na
          .sum(axis=0)
          .to_frame(name)
          .groupby(name)
          .size()
          .sort_index()
          .plot
          .line(style='-',
                ax=axes[1])
          )
    if min_samples_per_feat is not None:
        ax.vlines(min_samples_per_feat, *ax.get_ylim(), color='red')
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    ax.set_ylabel('observations (features)')
    fig.tight_layout()
    return fig


def plot_missing_dist_boxplots(data: pd.DataFrame,
                               min_feat_per_sample=None,
                               min_samples_per_feat=None) -> matplotlib.figure.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    not_na = data.notna()
    ax = (not_na
          .sum(axis=1)
          .rename('observation (feature) per sample')
          .plot
          .box(ax=axes[0])
          )
    if min_feat_per_sample is not None:
        ax.hlines(min_feat_per_sample, *ax.get_xlim(), color='red')
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    ax = (not_na
          .sum(axis=0)
          .rename('observation (samples) per feature')
          .plot
          .box(ax=axes[1])
          )
    if min_samples_per_feat is not None:
        ax.hlines(min_samples_per_feat, *ax.get_xlim(), color='red')
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    return fig


def plot_missing_pattern_violinplot(data: pd.DataFrame,
                                    min_feat_per_sample=None,
                                    min_samples_per_feat=None) -> matplotlib.figure.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    not_na = data.notna()
    name = 'features per sample'
    ax = sns.violinplot(
        data=not_na.sum(axis=1).to_frame(name),
        ax=axes[0],
    )
    if min_feat_per_sample is not None:
        ax.hlines(min_feat_per_sample, *ax.get_xlim(), color='red')
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    ax.set_ylabel('observations (features)')
    name = 'samples per feature'
    ax = sns.violinplot(
        data=not_na.sum(axis=0).to_frame(name),
        ax=axes[1],
    )
    if min_samples_per_feat is not None:
        ax.hlines(min_samples_per_feat, *ax.get_xlim(), color='red')
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    ax.set_ylabel('observations (samples)')
    fig.tight_layout()
    return fig


def plot_missing_pattern_histogram(data: pd.DataFrame,
                                   bins: int = 20,
                                   min_feat_per_sample=None,
                                   min_samples_per_feat=None,) -> matplotlib.figure.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    not_na = data.notna()

    name = 'features per sample'
    ax = not_na.sum(axis=1).to_frame(name).plot.hist(
        ax=axes[0],
        bins=bins,
        legend=False,
    )
    if min_feat_per_sample is not None:
        ax.vlines(min_feat_per_sample, *ax.get_ylim(), color='red')
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    ax.set_xlabel('observations (features)')
    ax.set_ylabel('observations')
    name = 'samples per feature'
    ax = data = not_na.sum(axis=0).to_frame(name).plot.hist(
        ax=axes[1],
        bins=bins,
        legend=False,
    )
    if min_samples_per_feat is not None:
        ax.vlines(min_samples_per_feat, *ax.get_ylim(), color='red')
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    ax.set_xlabel('observations (samples)')
    ax.set_ylabel(None)
    fig.tight_layout()
    return fig
