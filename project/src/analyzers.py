from types import SimpleNamespace

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.decomposition import PCA

from vaep.pandas import _add_indices

class Analysis(SimpleNamespace):

    def __repr__(self):
        ret = super().__repr__()
        return ret


class AnalyzePeptides(SimpleNamespace):
    """Namespace for current analysis

    Attributes
    ----------
    df:  pandas.DataFrame
        current eagerly loaded data
    stats: types.SimpleNamespace
        Some statistics of certain aspects. Normally each will be a DataFrame.

    Many more attributes are set dynamically depending on the concrete analysis.
    """

    def __init__(self, fname, nrows=None):
        self.df = self.read_csv(fname, nrows=nrows)
        self.stats = SimpleNamespace()
        self.is_log_transformed = False

    def read_csv(self, fname, nrows):
        return pd.read_csv(fname, index_col=0, low_memory=False, nrows=nrows)

    def describe_peptides(self, sample_n: int = None):
        if sample_n:
            df = self.df.sample(n=sample_n, axis=1)
        else:
            df = self.df
        stats = df.describe()
        stats.loc['CV'] = stats.loc['std'] / stats.loc['mean']
        self.stats.peptides = stats
        return stats

    def log_transform(self, log_fct:np.ufunc):
        """Log transform data in-place.

        Parameters
        ----------
        log_fct : np.ufunc
            Numpy log-function

        Raises
        ------
        Exception
            if data has been previously log-transformed.
        """
        if self.is_log_transformed:
            raise Exception(
                f'Data was already log transformed, using {self.__class__.__name__}.log_fct: {self.log_fct}')
        else:
            self.df = log_fct(self.df)
            self.is_log_transformed = True
            self.log_fct = log_fct
            

    def get_dectection_limit(self):
        """Compute information on detection limit in dataset.

        Returns
        -------
        str
            Information on detection limit
        """
        self.detection_limit = self.df.min().min() if self.is_log_transformed else np.log10(
            self.df).min().min()  # all zeros become nan.
        return "Detection limit: {:6.3f}, corresponding to intensity value of {:,d}".format(
            self.detection_limit,
            int(10 ** self.detection_limit))

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}".format(k, self.__dict__[k]) for k in keys)
        return "{} with attributes: {}".format(type(self).__name__, ", ".join(items))


def corr_lower_triangle(df):
    """Compute the correlation matrix, returning only unique values."""
    corr_df = df.corr()
    lower_triangle = pd.DataFrame(
        np.tril(np.ones(corr_df.shape), -1)).astype(bool)
    lower_triangle.index, lower_triangle.columns = corr_df.index, corr_df.columns
    return corr_df.where(lower_triangle)


def plot_corr_histogram(corr_lower_triangle, bins=10):
    fig, axes = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [
        5, 1], "wspace": 0.2}, figsize=(10, 4))
    values = pd.Series(corr_lower_triangle.to_numpy().flatten()).dropna()
    ax = axes[0]
    values.hist(ax=ax, bins=bins)
    ax = axes[1]
    plt.axis('off')
    data = values.describe().round(2)
    data.name = ''
    _ = pd.plotting.table(ax=ax, data=data, loc="best", edges="open")
    return fig, axes


def run_pca(df, n_components=2):
    """Run PCA on DataFrame.

    Returns
    -------
    pandas.DataFrame
        with same indices as in original DataFrame
    """
    pca = PCA(n_components=n_components).fit_transform(df)
    cols = [f'pc{i}' for i in range(n_components)]
    pca = pd.DataFrame(pca, index=df.index, columns=cols)
    return pca


def scatter_plot_w_dates(ax, df, dates=None, errors='raise'):
    """plot first vs. second column in DataFrame.
    Use dates to color data.
    
    
    
    errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        Passed on to pandas.to_datetime
        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaT.
        - If 'ignore', then invalid parsing will return the input.
    """
    # Inspiration:  https://stackoverflow.com/a/59685599/9684872
    cols = df.columns

    if isinstance(dates, str):
        dates = df['dates']

    path_collection = ax.scatter(
        x=df[cols[0]],
        y=df[cols[1]],
        c=[mdates.date2num(t) for t in pd.to_datetime(dates, errors=errors)
           ] if dates is not None else None
    )
    return path_collection

def add_date_colorbar(ax, fig):
    loc = mdates.AutoDateLocator()
    _ = fig.colorbar(ax, ticks=loc,
                 format=mdates.AutoDateFormatter(loc))
    return ax