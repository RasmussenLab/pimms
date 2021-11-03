from collections import namedtuple
from operator import index
from types import SimpleNamespace
import itertools
import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFwe
from sklearn.impute import SimpleImputer


from vaep.pandas import _add_indices

from . import metadata

ALPHA = 0.5


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

    def __init__(self, fname, nrows=None, index_col='Sample ID'):
        self.df = read_csv(fname, nrows=nrows, index_col=index_col)
        self.N, self.M = self.df.shape
        assert f'N{self.N:05d}' in str(fname) and f'M{self.M:05d}' in str(fname), \
            f"Filename number don't match loaded numbers: {fname} should contain N{self.N} and M{self.M}"
        self.stats = SimpleNamespace()
        self.is_log_transformed = False
        self.is_wide_format = True
        self.index_col = index_col

    def get_consecutive_dates(self, n_samples, seed=42):
        """Select n consecutive samples using a seed.
        
        Updated the original DataFrame attribute: df
        """
        self.df.sort_index(inplace=True)
        n_samples = min(len(self.df), n_samples) if n_samples else len(self.df)
        print(f"Get {n_samples} samples.")

        if seed:
            random.seed(42)

        _attr_name = f'df_{n_samples}'
        setattr(self, _attr_name, get_consecutive_data_indices(self.df, n_samples))
        print("Training data referenced unter:", _attr_name)
        self.df = getattr(self, _attr_name)
        print("Updated attribute: df")
        return self.df

    @property
    def df_long(self):
        if hasattr(self, '_df_long'):
            return self._df_long
        return self.to_long_format(colname_values='intensity', index_name=self.index_col)

    def to_long_format(self, colname_values: str = 'intensity', index_name: str = 'Sample ID', inplace: str = False) -> pd.DataFrame:
        """[summary]

        Parameters
        ----------
        colname_values : str, optional
            New column name for values in matrix, by default 'intensity'
        index_name : str, optional
            Name of column to assign as index (based on long-data format), by default 'Sample ID'
        inplace : bool, optional
            Assign result to df_long (False), or to df (True) attribute, by default False
        Returns
        -------
        pd.DataFrame
            Data in long-format as DataFrame
        """

        """Build long data view."""
        if not self.is_wide_format:
            return self.df
        if hasattr(self, '_df_long'):
            return self._df_long  # rm attribute to overwrite

        df_long = long_format(
            self.df, colname_values=colname_values, index_name=index_name)

        if inplace:
            self.df = df_long
            self.is_wide_format = False
            return self.df
        self._df_long = df_long
        return df_long

    @property
    def df_wide(self):
        return self.to_wide_format()

    def to_wide_format(self, columns: str = 'Sample ID', name_values: str = 'intensity', inplace: bool = False) -> pd.DataFrame:
        """[summary]

        Parameters
        ----------
        columns : str, optional
            Index level to be shown as columns, by default 'Sample ID'
        name_values : str, optional
            Column in long-data format to be used as values, by default 'intensity'
        inplace : bool, optional
            Assign result to df_wide (False), or to df (True) attribute, by default False

        Returns
        -------
        pd.DataFrame
            [description]
        """

        """Build wide data view.
        
        Return df attribute in case this is in wide-format. If df attribute is in long-format
        this is used. If df is wide, but long-format exist, then the wide format is build.
        
        
        """
        if self.is_wide_format:
            return self.df

        if hasattr(self, '_df_long'):
            df = self._df_long
        else:
            df = self.df

        df_wide = wide_format(df, columns=columns, name_values=name_values)

        if inplace:
            self.df = df_wide
            self.is_wide_format = True
            return self.df
        self._df_wide = df_wide
        print(f"Set attribute: df_wide")
        return df_wide

    def describe_peptides(self, sample_n: int = None):
        if sample_n:
            df = self.df.sample(n=sample_n, axis=1)
        else:
            df = self.df
        stats = df.describe()
        stats.loc['CV'] = stats.loc['std'] / stats.loc['mean']
        self.stats.peptides = stats
        return stats

    def get_prop_not_na(self):
        """Get prop. of not NA values for each sample."""
        return self.df.notna().sum(axis=1) / self.df.shape[-1]

    def add_metadata(self, add_prop_not_na=True):
        d_meta = metadata.get_metadata_from_filenames(self.df.index)
        self.df_meta = pd.DataFrame.from_dict(
            d_meta, orient='index')
        self.df_meta.index.name = self.df.index.name
        print(f'Created metadata DataFrame attribute `df_meta`.')
        # add proportion on not NA to meta data
        if add_prop_not_na:
            self.df_meta['prop_not_na'] = self.get_prop_not_na()
        print(f'Added proportion of not NA values based on `df` intensities.')
        return self.df_meta

    def get_PCA(self, n_components=2, imputer=SimpleImputer):
        X = imputer().fit_transform(self.df)
        X = _add_indices(X, self.df)
        assert all(X.notna())

        pca = run_pca(X, n_components=n_components)
        if not hasattr(self, 'df_meta'):
            _ = self.add_metadata()
        pca['ms_instrument'] = self.df_meta['ms_instrument'].astype('category')
        return pca

    def plot_pca(self,):
        """Create principal component plot with three heatmaps showing
        instrument, degree of non NA data and sample by date."""
        if not hasattr(self, 'df_meta'):
            _ = self.add_metadata()

        pca = self.get_PCA()
        cols = list(pca.columns)

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(
            15, 20), constrained_layout=True)

        Dim = namedtuple('DimensionsData', 'N M')
        self.dim = Dim(*self.df.shape)

        fig.suptitle(
            f'First two Principal Components of {self.dim.M} most abundant peptides \n for {self.dim.N} samples', fontsize=30)

        # by instrument
        ax = axes[0]
        # for name, group in pca.groupby('ms_instrument'):
        #     ax.scatter(x=group[cols[0]], y=group[cols[1]], label=name)
        seaborn.scatterplot(x=pca[cols[0]], y=pca[cols[1]],
                            hue=pca['ms_instrument'], ax=ax, palette='deep')
        ax.set_title('by category', fontsize=18)
        ax.legend(loc='center right', bbox_to_anchor=(1.11, 0.5))

        # by complettness/missingness
        # continues colormap will be a bit trickier using seaborn: https://stackoverflow.com/a/44642014/9684872
        ax = axes[1]
        ax.set_title('by number on na', fontsize=18)
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        path_collection = ax.scatter(
            x=cols[0], y=cols[1], c=self.df_meta['prop_not_na'], data=pca, alpha=ALPHA)
        _ = fig.colorbar(path_collection, ax=ax)

        # by dates
        ax = axes[2]
        ax.set_title('by date', fontsize=18)
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        path_collection = scatter_plot_w_dates(
            ax, pca, dates=self.df_meta.date, errors='raise')
        path_collection = add_date_colorbar(path_collection, ax=ax, fig=fig)
        return fig

    def log_transform(self, log_fct: np.ufunc):
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

    @property
    def fname_stub(self):
        assert hasattr(self, 'df'), f'Attribute df is missing: {self}'
        return 'N{:05d}_M{:05d}'.format(*self.df.shape)


def read_csv(fname, nrows, index_col=None):
    return pd.read_csv(fname, index_col=index_col, low_memory=False, nrows=nrows)


def get_consecutive_data_indices(df, n_samples):
    index = df.sort_index().index
    start_sample = len(index) - n_samples
    start_sample = random.randint(0, start_sample)
    return df.loc[index[start_sample:start_sample+n_samples]]


def long_format(df: pd.DataFrame,
                colname_values: str = 'intensity',
                index_name: str = 'Sample ID') -> pd.DataFrame:
    # ToDo: Docstring as in class when finalized
    df_long = df.unstack().dropna().to_frame(colname_values)
    df_long = df_long.reset_index('Sample ID')
    return df_long


def wide_format(df: pd.DataFrame,
                columns: str = 'Sample ID',
                name_values: str = 'intensity') -> pd.DataFrame:
    # ToDo: Docstring as in class when finalized
    df_wide = df.pivot(columns=columns, values=name_values)
    df_wide = df_wide.T
    return df_wide


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
    pca = PCA(n_components=n_components)
    PCs = pca.fit_transform(df)
    cols = [f'principal component {i+1} ({var_explained*100:.2f} %)' for i,
            var_explained in enumerate(pca.explained_variance_ratio_)]
    pca = pd.DataFrame(PCs, index=df.index, columns=cols)
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
           ] if dates is not None else None,
        alpha=ALPHA
    )
    return path_collection


def add_date_colorbar(mappable, ax, fig):
    loc = mdates.AutoDateLocator()
    _ = fig.colorbar(mappable, ax=ax, ticks=loc,
                     format=mdates.AutoDateFormatter(loc))
    return ax


def cast_object_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """Cast object columns to category dtype.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns

    Returns
    -------
    pd.DataFrame
        DataFrame with category columns instead of object columns.
    """
    _columns = df.select_dtypes(include='object').columns
    return df.astype({col: 'category' for col in _columns})

