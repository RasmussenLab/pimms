from __future__ import annotations

import logging
import pathlib
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

import vaep.pandas
from vaep.plotting import data, defaults, errors, plotly
from vaep.plotting.errors import plot_rolling_error

seaborn.set_style("whitegrid")
# seaborn.set_theme()

plt.rcParams['figure.figsize'] = [16.0, 7.0]  # [4, 2], [4, 3]
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['figure.dpi'] = 147


logger = logging.getLogger(__name__)

__all__ = ['plotly',
           'data',
           'defaults',
           'errors',
           'plot_rolling_error',
           # define in this file
           'savefig',
           'select_xticks',
           'select_dates',
           'make_large_descriptors',
           'plot_feat_counts',
           'plot_cutoffs',
           ]


def _savefig(fig, name, folder: pathlib.Path = '.',
             pdf=True,
             dpi=300,  # default 'figure',
             tight_layout=True,
             ):
    """Save matplotlib Figure (having method `savefig`) as pdf and png."""
    folder = pathlib.Path(folder)
    fname = folder / name
    folder = fname.parent  # in case name specifies folders
    folder.mkdir(exist_ok=True, parents=True)
    if tight_layout:
        fig.tight_layout()
    fig.savefig(fname.with_suffix('.png'), dpi=dpi)
    if pdf:
        fig.savefig(fname.with_suffix('.pdf'), dpi=dpi)
    logger.info(f"Saved Figures to {fname}")


savefig = _savefig


def select_xticks(ax: matplotlib.axes.Axes, max_ticks: int = 50) -> list:
    """Limit the number of xticks displayed.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to manipulate
    max_ticks : int, optional
        maximum number of set ticks on x-axis, by default 50

    Returns
    -------
    list
        list of current ticks for x-axis. Either new
        or old (depending if something was changed).
    """
    x_ticks = ax.get_xticks()
    offset = len(x_ticks) // max_ticks
    if offset > 1:  # if larger than 1
        return ax.set_xticks(x_ticks[::offset])
    return x_ticks


def select_dates(date_series: pd.Series, max_ticks=30) -> np.array:
    """Get unique dates (single days) for selection in pd.plot.line
    with xticks argument.

    Parameters
    ----------
    date_series : pd.Series
        datetime series to use (values, not index)
    max_ticks : int, optional
        maximum number of unique ticks to select, by default 30

    Returns
    -------
    np.array
        _description_
    """
    xticks = date_series.dt.date.unique()
    offset = len(xticks) // max_ticks
    if offset > 1:
        return xticks[::offset]
    else:
        xticks


def make_large_descriptors(size='xx-large'):
    """Helper function to have very large titles, labes and tick texts for
    matplotlib plots per default.

    size: str
        fontsize or allowed category. Change default if necessary, default 'xx-large'
    """
    plt.rcParams.update({k: size for k in ['xtick.labelsize',
                                           'ytick.labelsize',
                                           'axes.titlesize',
                                           'axes.labelsize',
                                           'legend.fontsize',
                                           'legend.title_fontsize']
                         })


set_font_sizes = make_large_descriptors


def add_prop_as_second_yaxis(ax: matplotlib.axes.Axes, n_samples: int,
                             format_str: str = '{x:,.3f}') -> matplotlib.axes.Axes:
    """Add proportion as second axis. Try to align cleverly

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes for which you want to add a second y-axis
    n_samples : int
        Number of total samples (to normalize against)

    Returns
    -------
    matplotlib.axes.Axes
        Second layover twin Axes with right-hand side y-axis
    """
    ax2 = ax.twinx()
    n_min, n_max = np.round(ax.get_ybound())
    logger.info(f"{n_min = }, {n_max = }")
    lower_prop = n_min / n_samples + (ax.get_ybound()[0] - n_min) / n_samples
    upper_prop = n_max / n_samples + (ax.get_ybound()[1] - n_max) / n_samples
    logger.info(f'{lower_prop = }, {upper_prop = }')
    ax2.set_ybound(lower_prop, upper_prop)
    # _ = ax2.set_yticks(np.linspace(n_min/n_samples,
    #                    n_max /n_samples, len(ax.get_yticks())-2))
    _ = ax2.set_yticks(ax.get_yticks()[1:-1] / n_samples)
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.StrMethodFormatter(format_str))
    return ax2


def add_height_to_barplot(ax, size=5, rotated=False):
    ax.annotate = partial(ax.annotate, text='NA',
                          xytext=(0, int(size / 2)),
                          ha='center',
                          size=size,
                          textcoords='offset points')
    ax.annotate = partial(ax.annotate,
                          rotation=0,
                          va='center')
    if rotated:
        ax.annotate = partial(ax.annotate,
                              xytext=(1, int(size / 3)),
                              rotation=90,
                              va='bottom')
    for bar in ax.patches:
        if not bar.get_height():
            xy = (bar.get_x() + bar.get_width() / 2,
                  0.0)
            ax.annotate(text='NA',
                        xy=xy,
                        )
            continue
        ax.annotate(text=format(bar.get_height(), '.2f'),
                    xy=(bar.get_x() + bar.get_width() / 2,
                        bar.get_height()),
                    )
    return ax


def add_text_to_barplot(ax, text, size=5):
    for bar, text_ in zip(ax.patches, text):
        logger.debug(f"{bar = }, f{text = }, {bar.get_height() = }")
        if not bar.get_height():
            continue
        ax.annotate(text=text_,
                    xy=(bar.get_x() + bar.get_width() / 2,
                        bar.get_height()),
                    xytext=(1, -5),
                    rotation=90,
                    ha='center',
                    va='top',
                    size=size,
                    textcoords='offset points')
    return ax


def format_large_numbers(ax: matplotlib.axes.Axes,
                         format_str: str = '{x:,.0f}') -> matplotlib.axes.Axes:
    """Format large integer numbers to be read more easily.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes which labels should be manipulated.
    format_str : str, optional
        Default float format string, by default '{x:,.0f}'

    Returns
    -------
    matplotlib.axes.Axes
        _description_
    """
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.StrMethodFormatter(format_str))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.StrMethodFormatter(format_str))
    return ax


def plot_feat_counts(df_counts: pd.DataFrame, feat_name: str, n_samples: int,
                     ax=None, figsize=(15, 10),
                     count_col='counts',
                     **kwargs):
    args = dict(
        ylabel='count',
        xlabel=f'{feat_name} ordered by completeness',
        title=f'Count and proportion of {len(df_counts):,d} {feat_name}s over {n_samples:,d} samples',
    )
    args.update(kwargs)

    ax = df_counts[count_col].plot(
        figsize=figsize,

        grid=True,
        ax=ax,
        **args)

    # default nearly okay, but rather customize to see minimal and maxium proportion
    # ax = peptide_counts['proportion'].plot(secondary_y=True, style='b')

    ax2 = add_prop_as_second_yaxis(ax=ax, n_samples=n_samples)
    ax2.set_ylabel('proportion')
    ax = format_large_numbers(ax=ax)
    return ax


def plot_counts(df_counts: pd.DataFrame, n_samples,
                feat_col_name: str = 'count',
                feature_name=None,
                ax=None, prop_feat=0.25, min_feat_prop=.01,
                **kwargs):
    """Plot counts based on get_df_counts."""
    if feature_name is None:
        feature_name = feat_col_name
    # df_counts = df_counts[[feat_col_name]].copy()
    ax = plot_feat_counts(df_counts,
                          feat_name=feature_name,
                          n_samples=n_samples,
                          count_col=feat_col_name,
                          ax=ax, **kwargs)
    df_counts['prop'] = df_counts[feat_col_name] / n_samples
    n_feat_cutoff = vaep.pandas.get_last_index_matching_proportion(
        df_counts=df_counts, prop=prop_feat, prop_col='prop')
    n_samples_cutoff = df_counts.loc[n_feat_cutoff, feat_col_name]
    logger.info(f'{n_feat_cutoff = }, {n_samples_cutoff = }')
    x_lim_max = vaep.pandas.get_last_index_matching_proportion(
        df_counts, min_feat_prop, prop_col='prop')
    logger.info(f'{x_lim_max = }')
    ax.set_xlim(-1, x_lim_max)
    ax.axvline(n_feat_cutoff, c='red')

    # ax.text(n_feat_cutoff + 0.03 * x_lim_max,
    #         n_samples_cutoff, '25% cutoff',
    #         style='italic', fontsize=12,
    #         bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})

    ax.annotate(f'{prop_feat*100}% cutoff',
                xy=(n_feat_cutoff, n_samples_cutoff),
                xytext=(n_feat_cutoff + 0.1 * x_lim_max, n_samples_cutoff),
                fontsize=16,
                arrowprops=dict(facecolor='black', shrink=0.05))
    return ax


def plot_cutoffs(df: pd.DataFrame,
                 feat_completness_over_samples: int = None,
                 min_feat_in_sample: int = None
                 ) -> tuple[matplotlib.figure.Figure, np.array[matplotlib.axes.Axes]]:
    """plot number of available features along index and columns (feat vs samples),
    potentially including some cutoff.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame in wide data format.
    feat_completness_over_samples : int, optional
        horizental line to plot as cutoff for features, by default None
    min_feat_in_sample : int, optional
        horizental line to plot as cutoff for samples, by default None

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.array[matplotlib.axes.Axes]]
        _description_
    """
    notna = df.notna()
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    notna.sum(axis=0).sort_values().plot(rot=90, ax=ax,
                                         ylabel='count samples', xlabel='feature name')
    if feat_completness_over_samples is not None:
        ax.axhline(feat_completness_over_samples)
    ax = axes[1]
    notna.sum(axis=1).sort_values().plot(rot=90, ax=ax,
                                         ylabel='count features', xlabel='sample name')
    if min_feat_in_sample is not None:
        ax.axhline(min_feat_in_sample)
    return fig, axes


def only_every_x_ticks(ax, x=2, axis=None):
    """Sparse out ticks on both axis by factor x"""
    if axis is None:
        ax.set_xticks(ax.get_xticks()[::x])
        ax.set_yticks(ax.get_yticks()[::x])
    else:
        if axis == 0:
            ax.set_xticks(ax.get_xticks()[::x])
        elif axis == 1:
            ax.set_yticks(ax.get_yticks()[::x])
        else:
            raise ValueError(f'axis must be 0 or 1, got {axis}')
    return ax


def use_first_n_chars_in_labels(ax, x=2):
    """Take first N characters of labels and use them as new labels"""
    # xaxis
    _new_labels = [_l.get_text()[:x]
                   for _l in ax.get_xticklabels()]
    _ = ax.set_xticklabels(_new_labels)
    # yaxis
    _new_labels = [_l.get_text()[:x] for _l in ax.get_yticklabels()]
    _ = ax.set_yticklabels(_new_labels)
    return ax


def split_xticklabels(ax, PG_SEPARATOR=';'):
    """Split labels by PG_SEPARATOR and only use first part"""
    if PG_SEPARATOR is not None:
        _new_labels = [_l.get_text().split(PG_SEPARATOR)[0]
                       for _l in ax.get_xticklabels()]
        _ = ax.set_xticklabels(_new_labels)
    return ax
