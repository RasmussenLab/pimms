"""Plot data distribution based on pandas `DataFrames` or `Series`."""
import logging
from typing import Iterable, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def min_max(s: pd.Series) -> Tuple[int]:
    """Get the min and max as integer from a pandas.Series.

    Parameters
    ----------
    s : pd.Series
        Series of intensities.

    Returns
    -------
    Tuple[int]
        _description_
    """
    min_bin, max_bin = (int(s.min()), (int(s.max()) + 1))
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


def plot_histogram_intensities(s: pd.Series,
                               interval_bins=1,
                               min_max: Tuple[int] = None,
                               ax=None,
                               **kwargs) -> Tuple[Axes, range]:
    """Plot intensities in Series in a certain range and equally spaced intervals."""
    if min_max is None:
        min_max = get_min_max_iterable([s])
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
                      size: int = 1,
                      ylabel: str = 'Frequency',
                      xlabel: Optional[str] = None) -> Axes:
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
    if xlabel is None:
        if df.columns.name:
            xlabel = f'Samples ordered by identified {df.columns.name}'
        else:
            xlabel = 'Samples ordered by identified features'

    ax = (df
          .notna()
          .sum(axis=axis)
          .sort_values()
          .reset_index(drop=True)
          .plot(
              ax=ax,
              style='.',
              ms=size,
              title=title,
              ylabel=ylabel,
              xlabel=xlabel)
          )
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    return ax


def plot_missing_dist_highdim(data: pd.DataFrame,
                              min_feat_per_sample: int = None,
                              min_samples_per_feat: int = None) -> matplotlib.figure.Figure:
    """Plot missing distribution (cdf) in high dimensional data.

    Parameters
    ----------
    data : pd.DataFrame
        Intensity table with samples in rows and features in columns.
    min_feat_per_sample : int, optional
        Show the minimum required features a sample has to have, by default None
    min_samples_per_feat : int, optional
        Show the minimum required number of samples a feature has to be found in, by default None

    Returns
    -------
    matplotlib.figure.Figure
        Figure with two plots (Axes).
    """
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
          .line(style='.',
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
          .line(style='.',
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
    idx_label, col_label = 'feature', 'sample'
    if data.index.name:
        idx_label = data.index.name
    if data.columns.name:
        col_label = data.columns.name
    ax = (not_na
          .sum(axis=1)
          .rename(f'observation ({idx_label}) per {col_label}')
          .plot
          .box(ax=axes[0])
          )
    if min_feat_per_sample is not None:
        ax.hlines(min_feat_per_sample, *ax.get_xlim(), color='red')
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    ax = (not_na
          .sum(axis=0)
          .rename(f'observation ({idx_label}) per {col_label}')
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
    idx_label, col_label = 'sample', 'feature'
    if data.index.name:
        idx_label = data.index.name
    if data.columns.name:
        col_label = data.columns.name
    name = f'observations per {idx_label}'
    ax = not_na.sum(axis=1).to_frame(name).plot.hist(
        ax=axes[0],
        bins=bins,
        legend=False,
    )
    if min_feat_per_sample is not None:
        ax.vlines(min_feat_per_sample, *ax.get_ylim(), color='red')
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    ax.set_xlabel(name)
    ax.set_ylabel('observations in bin')
    # second
    name = f'observations per {col_label}'
    ax = data = not_na.sum(axis=0).to_frame(name).plot.hist(
        ax=axes[1],
        bins=bins,
        legend=False,
    )
    if min_samples_per_feat is not None:
        ax.vlines(min_samples_per_feat, *ax.get_ylim(), color='red')
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    ax.set_xlabel(name)
    ax.set_ylabel(None)
    fig.tight_layout()
    return fig


def plot_feat_median_over_prop_missing(data: pd.DataFrame,
                                       type: str = 'scatter',
                                       ax: matplotlib.axes.Axes = None,
                                       s: int = 1,
                                       return_plot_data: bool = False
                                       ) -> Union[matplotlib.axes.Axes,
                                                  Tuple[matplotlib.axes.Axes, pd.DataFrame]]:
    """Plot feature median over proportion missing in that feature.
    Sorted by feature median into bins."""
    y_col = 'prop. missing'
    x_col = 'Features binned by their median intensity (N features)'

    missing_by_median = {
        'median feat value': data.median(),
        y_col: data.isna().mean()}
    missing_by_median = pd.DataFrame(missing_by_median)

    bins = range(
        *min_max(missing_by_median['median feat value']), 1)

    missing_by_median['bins'] = pd.cut(
        missing_by_median['median feat value'], bins=bins)
    missing_by_median['median feat value (floor)'] = (missing_by_median['median feat value']
                                                      .astype(int)
                                                      )
    _counts = (missing_by_median
               .groupby('median feat value (floor)')['median feat value']
               .count()
               .rename('count'))
    missing_by_median = missing_by_median.join(
        _counts, on='median feat value (floor)')
    missing_by_median = missing_by_median.sort_values(
        'median feat value (floor)')
    missing_by_median[x_col] = (missing_by_median.iloc[:, -2:]
                                .apply(lambda s: "{:02,d}  (N={:3,d})".format(*s), axis=1)
                                )
    if type == 'scatter':
        ax = missing_by_median.plot.scatter(x_col, y_col,
                                            ylim=(-.03, 1.03),
                                            ax=ax,
                                            s=s,)
        # # for some reason this does not work as it does elswhere:
        # _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # # do it manually:
        _ = [(_l.set_rotation(45), _l.set_horizontalalignment('right'))
             for _l in ax.get_xticklabels()]
    elif type == 'boxplot':
        ax = missing_by_median[[x_col, y_col]].plot.box(
            by=x_col,
            boxprops=dict(linewidth=s),
            flierprops=dict(markersize=s),
            ax=ax,
        )
        ax = ax[0]  # returned series due to by argument?
        _ = ax.set_title('')
        _ = ax.set_ylabel(y_col)
        _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                               horizontalalignment='right')
        _ = ax.set_xlabel(x_col)
        _ = ax.set_ylim(-0.03, 1.03)
    else:
        raise ValueError(
            f'Unknown plot type: {type}, choose from: scatter, boxplot')
    if return_plot_data:
        return ax, missing_by_median
    return ax
