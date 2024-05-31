import logging
import random
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from njab.sklearn import run_pca
from sklearn.impute import SimpleImputer

import vaep
from vaep.analyzers import Analysis
from vaep.io.datasplits import long_format, wide_format
from vaep.io.load import verify_df
from vaep.pandas import _add_indices

logger = logging.getLogger(__name__)

__doc__ = 'A collection of Analyzers to perform certain type of analysis.'


ALPHA = 0.5

# ! deprecate AnalyzePeptides


class AnalyzePeptides(SimpleNamespace):
    """Namespace for current analysis

    Attributes
    ----------
    df:  pandas.DataFrame
        current eagerly loaded data in wide format only: sample index, features in columns
    stats: types.SimpleNamespace
        Some statistics of certain aspects. Normally each will be a DataFrame.

    Many more attributes are set dynamically depending on the concrete analysis.
    """

    def __init__(self, data: pd.DataFrame,
                 is_log_transformed: bool = False,
                 is_wide_format: bool = True, ind_unstack: str = '',):
        if not is_wide_format:
            if not ind_unstack:
                raise ValueError("Please specify index level for unstacking via "
                                 f"'ind_unstack' from: {data.index.names}")
            data = data.unstack(ind_unstack)
            is_wide_format = True
        self.df = data  # assume wide
        self.N, self.M = self.df.shape
        self.stats = SimpleNamespace()
        self.is_log_transformed = is_log_transformed
        self.is_wide_format = is_wide_format
        self.index_col = self.df.index.name

    @classmethod
    def from_csv(cls, fname: str,
                 nrows: int = None,
                 # could be potentially 0 for the first column
                 index_col: Union[int, str, List] = 'Sample ID',
                 verify_fname: bool = False,
                 usecols=None,
                 **kwargs):
        df = pd.read_csv(fname, index_col=index_col, low_memory=False,
                         nrows=nrows, usecols=usecols).squeeze('columns')
        if len(df.shape) == 1:
            # unstack all but first column
            df = df.unstack(df.index.names[1:])
        verify_df(df=df, fname=fname,
                  index_col=index_col,
                  verify_fname=verify_fname,
                  usecols=usecols)
        return cls(data=df, **kwargs)  # all __init__ parameters are kwargs

    @classmethod
    # @delegates(from_csv)  # does only include parameters with defaults
    def from_pickle(cls, fname: str,
                    # could be potentially 0 for the first column
                    index_col: Union[int, str, List] = 'Sample ID',
                    verify_fname: bool = False,
                    usecols=None,
                    **kwargs):
        df = pd.read_pickle(fname).squeeze()
        if len(df.shape) == 1:
            df = df.unstack(df.index.names[1:])
        verify_df(df=df, fname=fname,
                  index_col=index_col,
                  verify_fname=verify_fname,
                  usecols=usecols)
        return cls(data=df, **kwargs)  # all __init__ parameters are kwargs

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

    def to_long_format(
            self,
            colname_values: str = 'intensity',
            index_name: str = 'Sample ID',
            inplace: str = False) -> pd.DataFrame:
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
            self.df,
            colname_values=colname_values,
            # index_name=index_name
        )

        if inplace:
            self.df = df_long
            self.is_wide_format = False
            return self.df
        self._df_long = df_long
        return df_long

    @property
    def df_wide(self):
        return self.to_wide_format()

    def to_wide_format(
            self,
            columns: str = 'Sample ID',
            name_values: str = 'intensity',
            inplace: bool = False) -> pd.DataFrame:
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
        print("Set attribute: df_wide")
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

    def get_PCA(self, n_components=2, imputer=SimpleImputer):
        self.imputer_ = imputer()
        X = self.imputer_.fit_transform(self.df)
        X = _add_indices(X, self.df)
        assert all(X.notna())

        PCs, self.pca_ = run_pca(X, n_components=n_components)
        if not hasattr(self, 'df_meta'):
            logger.warning('No metadata available, please set "df_meta" first.')
        try:
            PCs['ms_instrument'] = self.df_meta['ms_instrument'].astype('category')
        except KeyError:
            logger.warning("No MS instrument added.")
        except AttributeError:
            logger.warning("No metadata available, please set 'df_meta' first.")
            logger.warning("No MS instrument added.")
        return PCs

    def calculate_PCs(self, new_df, is_wide=True):
        if not is_wide:
            new_df = new_df.unstack(new_df.index.names[1:])

        X = self.imputer_.transform(new_df)
        X = _add_indices(X, new_df)
        PCs = self.pca_.transform(X)
        PCs = _add_indices(PCs, new_df, index_only=True)
        PCs.columns = [f'PC {i+1}' for i in range(PCs.shape[-1])]
        return PCs

    def plot_pca(self,):
        """Create principal component plot with three heatmaps showing
        instrument, degree of non NA data and sample by date."""
        if not self.is_wide_format:
            self.df = self.df.unstack(self.df.index.names[1:])
            self.is_wide_format = True

        if not hasattr(self, 'df_meta'):
            raise AttributeError('No metadata available, please set "df_meta" first.')

        PCs = self.get_PCA()

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(
            15, 20), constrained_layout=True)

        Dim = namedtuple('DimensionsData', 'N M')
        self.dim = Dim(*self.df.shape)

        fig.suptitle(
            f'First two Principal Components of {self.dim.M} most abundant peptides \n for {self.dim.N} samples',
            fontsize=30)

        # by instrument
        ax = axes[0]
        seaborn_scatter(df=PCs.iloc[:, :2], fig=fig, ax=ax,
                        meta=PCs['ms_instrument'], title='by MS instrument')
        ax.legend(loc='center right', bbox_to_anchor=(1.11, 0.5))

        # by complettness/missingness
        # continues colormap will be a bit trickier using seaborn: https://stackoverflow.com/a/44642014/9684872
        ax = axes[1]
        plot_scatter(df=PCs.iloc[:, :2], fig=fig, ax=ax,
                     meta=self.df_meta['prop_not_na'], title='by number on na')

        # by dates
        ax = axes[2]
        plot_date_map(df=PCs.iloc[:, :2],
                      ax=ax, dates=self.df_meta.date)

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
        items = ("{}".format(k) for k in keys)
        return "{} with attributes: {}".format(type(self).__name__, ", ".join(items))

    # def __dir__(self):
    #     return sorted(self.__dict__)

    @property
    def fname_stub(self):
        assert hasattr(self, 'df'), f'Attribute df is missing: {self}'
        return 'N{:05d}_M{:05d}'.format(*self.df.shape)


class LatentAnalysis(Analysis):

    def __init__(self, latent_space: pd.DataFrame, meta_data: pd.DataFrame, model_name: str,
                 fig_size: Tuple[int, int] = (15, 15), folder: Path = None):
        self.latent_space, self.meta_data = latent_space, meta_data
        self.fig_size, self.folder = fig_size, folder
        self.model_name = model_name
        self.folder = Path(self.folder) if self.folder else Path('.')
        assert len(
            self.latent_space.shape) == 2, "Expected a two dimensional DataFrame."
        self.latent_dim = self.latent_space.shape[-1]
        if self.latent_dim > 2:
            # pca, add option for different methods
            self.latent_reduced, self.pca_ = run_pca(self.latent_space)
        else:
            self.latent_reduced = self.latent_space

    def plot_by_date(self, meta_key: str = 'date', save: bool = True):
        fig, ax = self._plot(fct=plot_date_map, meta_key=meta_key, save=save)
        return fig, ax

    def plot_by_category(self, meta_key: str, save: bool = True):
        fig, ax = self._plot(fct=seaborn_scatter, meta_key=meta_key, save=save)
        return fig, ax

    def _plot(self, fct, meta_key: str, save: bool = True):
        try:
            meta_data = self.meta_data[meta_key]
        except KeyError:
            raise ValueError(f"Requested key: '{meta_key}' is not in available,"
                             f" use: {', '.join(x for x in self.meta_data.columns)}")
        fig, ax = plt.subplots(figsize=self.fig_size)
        _ = fct(df=self.latent_reduced, ax=ax,
                meta=meta_data.loc[self.latent_reduced.index],
                title=f'{self.model_name} latent space PCA of {self.latent_dim} dimensions by {meta_key}')
        if save:
            vaep.plotting._savefig(fig, name=f'{self.model_name}_latent_by_{meta_key}',
                                   folder=self.folder)
        return fig, ax


# def read_csv(fname:str, nrows:int, index_col:str=None)-> pd.DataFrame:
#     return pd.read_csv(fname, index_col=index_col, low_memory=False, nrows=nrows)


def get_consecutive_data_indices(df, n_samples):
    index = df.sort_index().index
    start_sample = len(index) - n_samples
    start_sample = random.randint(0, start_sample)
    return df.loc[index[start_sample:start_sample + n_samples]]


def corr_lower_triangle(df, **kwargs):
    """Compute the correlation matrix, returning only unique values.
    """
    corr_df = df.corr(**kwargs)
    lower_triangle = pd.DataFrame(
        np.tril(np.ones(corr_df.shape), -1)).astype(bool)
    lower_triangle.index, lower_triangle.columns = corr_df.index, corr_df.columns
    return corr_df.where(lower_triangle)


def plot_corr_histogram(corr_lower_triangle, bins=10):
    fig, axes = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [
        5, 1], "wspace": 0.2}, figsize=(8, 4))
    values = pd.Series(corr_lower_triangle.to_numpy().flatten()).dropna()
    ax = axes[0]
    ax = values.hist(ax=ax, bins=bins)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    ax = axes[1]
    plt.axis('off')
    data = values.describe(percentiles=np.linspace(0.1, 1, 10)).round(2)
    data.name = ''
    _ = pd.plotting.table(ax=ax, data=data, loc="best", edges="open")
    return fig, axes


def plot_date_map(df, ax,
                  dates: pd.Series = None,
                  meta: pd.Series = None,
                  title: str = 'by date',
                  fontsize=8,
                  size=2):
    if dates is not None and meta is not None:
        raise ValueError("Only set either dates or meta parameters.")
        # ToDo: Clean up arguments
    if dates is None:
        dates = meta
    cols = list(df.columns)
    assert len(cols) == 2, f'Please provide two dimensons, not {df.columns}'
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    path_collection = scatter_plot_w_dates(
        ax, df, dates=dates, errors='raise')
    _ = add_date_colorbar(path_collection, ax=ax)


def plot_scatter(df, ax,
                 meta: pd.Series,
                 feat_name_display: str = 'features',
                 title: Optional[str] = None,
                 alpha=ALPHA,
                 fontsize=8,
                 size=2):
    cols = list(df.columns)
    assert len(cols) == 2, f'Please provide two dimensons, not {df.columns}'
    if not title:
        title = f'by identified {feat_name_display}'
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    path_collection = ax.scatter(
        x=cols[0], y=cols[1], s=size, c=meta, data=df, alpha=alpha)
    _ = ax.get_figure().colorbar(path_collection, ax=ax,
                                 label=f'Identified {feat_name_display}',
                                 #  ticklocation='left', # ignored by matplotlib
                                 location='right',  # ! left does not put colobar without overlapping y ticks
                                 format="{x:,.0f}",
                                 )


def seaborn_scatter(df, ax,
                    meta: pd.Series,
                    title: str = 'by some metadata',
                    alpha=ALPHA,
                    fontsize=5,
                    size=5):
    cols = list(df.columns)
    assert len(cols) == 2, f'Please provide two dimensons, not {df.columns}'
    seaborn.scatterplot(x=df[cols[0]], y=df[cols[1]],
                        hue=meta, ax=ax, palette='deep', s=size, alpha=alpha)
    _ = ax.legend(fontsize=fontsize,
                  title_fontsize=fontsize,
                  markerscale=0.4,
                  title=meta.name,
                  )
    ax.set_title(title, fontsize=fontsize)
    return ax


def scatter_plot_w_dates(ax, df,
                         dates=None,
                         marker=None,
                         errors='raise',
                         size=2):
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
        alpha=ALPHA,
        s=size,
        marker=marker,
    )
    return path_collection


def add_date_colorbar(mappable, ax):
    loc = mdates.AutoDateLocator()
    cbar = ax.get_figure().colorbar(mappable, ax=ax, ticks=loc,
                                    format=mdates.AutoDateFormatter(loc))
    return cbar


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
