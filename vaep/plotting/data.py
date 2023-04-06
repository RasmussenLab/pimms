"""Plot data distribution based on pandas DataFrames or Series."""
from typing import Tuple

from matplotlib.axes import Axes
import pandas as pd


# %%
def min_max(s: pd.Series) -> Tuple[int]:
    min_bin, max_bin = (int(s.min()), (int(s.max())+1))
    return min_bin, max_bin


def plot_histogram_intensites(s: pd.Series, interval_bins=1, min_max=(15, 40), ax=None) -> Tuple[Axes, range]:
    min_bin, max_bin = min_max
    bins = range(min_bin, max_bin, interval_bins)
    ax = s.plot.hist(bins=bins, xticks=list(bins), ax=ax)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    return ax, bins


def plot_observations(df: pd.DataFrame,
                      ax:Axes=None,
                      title: str = '',
                      axis: int = 1,
                      ylabel: str = 'number of features',
                      xlabel: str = 'Samples ordered by number of features') -> Axes:
    """Plot non missing observations by row (axis=1) or column (axis=0) in
    order of number of available observations.

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
    return ax
