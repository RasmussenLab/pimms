import pathlib
import logging

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _savefig(fig, name, folder: pathlib.Path = '.', pdf=True):
    """Save matplotlib Figure (having method `savefig`) as pdf and png."""
    folder = pathlib.Path(folder)
    fname = folder / name
    folder = fname.parent  # in case name specifies folders
    folder.mkdir(exist_ok=True, parents=True)
    fig.savefig(fname.with_suffix('.png'))
    if pdf:
        fig.savefig(fname.with_suffix('.pdf'))
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


def make_large_descriptors():
    """Helper function to have very large titles, labes and tick texts for 
    matplotlib plots per default."""
    plt.rcParams.update({'xtick.labelsize': 'xx-large',
                         'ytick.labelsize': 'xx-large',
                         'axes.titlesize':  'xx-large',
                         'axes.labelsize':  'xx-large',
                         })
