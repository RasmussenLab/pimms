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
