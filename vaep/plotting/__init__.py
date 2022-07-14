import numpy as np
import pandas as pd
import matplotlib
import logging
import pathlib
import matplotlib.pyplot as plt
import seaborn

plt.rcParams['figure.figsize'] = [16.0, 7.0]
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

seaborn.set_theme()


labels_dict = {"NA not interpolated valid_collab collab MSE": 'MSE',
               'batch_size_collab': 'bs',
               'n_hidden_layers': "No. of hidden layers",
               'latent_dim': 'hidden layer dimension',
               'subset_w_N': 'subset',
               'n_params': 'no. of parameter',
               "metric_value": 'value',
               'metric_name': 'metric',
               }

order_categories = {'data level': ['proteinGroups', 'aggPeptides', 'evidence'],
                    'model': ['median', 'interpolated', 'collab', 'DAE', 'VAE']}

IDX_ORDER = (['proteinGroups', 'aggPeptides', 'evidence'],
             ['median', 'interpolated', 'collab', 'DAE', 'VAE'])


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
    lower_prop = n_min/n_samples + (ax.get_ybound()[0] - n_min) / n_samples
    upper_prop = n_max/n_samples + (ax.get_ybound()[1] - n_max) / n_samples
    logger.info(f'{lower_prop = }, {upper_prop = }')
    ax2.set_ybound(lower_prop, upper_prop)
    # _ = ax2.set_yticks(np.linspace(n_min/n_samples,
    #                    n_max /n_samples, len(ax.get_yticks())-2))
    _ = ax2.set_yticks(ax.get_yticks()[1:-1]/n_samples)
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.StrMethodFormatter(format_str))
    return ax2


def add_height_to_barplot(ax, size=15):
    for bar in ax.patches:
        ax.annotate(text=format(bar.get_height(), '.2f'),
                    xy=(bar.get_x() + bar.get_width() / 2,
                        bar.get_height()),
                    xytext=(0, 7),
                    ha='center',
                    va='center',
                    size=size,
                    textcoords='offset points')
    return ax


def add_text_to_barplot(ax, text, size=15):
    for bar, text in zip(ax.patches, text):
        logger.debug(f"{bar = }, f{text = }, {bar.get_height() = }")
        ax.annotate(text=text,
                    xy=(bar.get_x() + bar.get_width() / 2,
                        bar.get_height()),
                    xytext=(0, -5),
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
