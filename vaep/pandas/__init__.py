import collections.abc
from collections import namedtuple
from types import SimpleNamespace
from typing import Iterable, List, Optional

import numpy as np
import omegaconf
import pandas as pd

from vaep.pandas.calc_errors import calc_errors_per_feat, get_absolute_error

__all__ = [
    'calc_errors_per_feat',
    'get_absolute_error',
    'unique_cols',
    'get_unique_non_unique_columns',
    'prop_unique_index',
    'replace_with',
    'index_to_dict',
    'get_columns_accessor',
    'get_columns_accessor_from_iterable',
    'select_max_by',
    'get_columns_namedtuple',
    'highlight_min',
    '_add_indices',
    'interpolate',
    'flatten_dict_of_dicts',
    'key_map',
    'parse_query_expression',
    'length',
    'get_last_index_matching_proportion',
    'get_lower_whiskers',
    'get_counts_per_bin']


def unique_cols(s: pd.Series) -> bool:
    """Check all entries are equal in pandas.Series

    Ref: https://stackoverflow.com/a/54405767/968487

    Parameters
    ----------
    s : pandas.Series
        Series to check uniqueness

    Returns
    -------
    bool
        Boolean on if all values are equal.
    """
    return (s.iloc[0] == s).all()


def get_unique_non_unique_columns(df: pd.DataFrame) -> SimpleNamespace:
    """Get back a namespace with an column.Index both
    of the unique and non-unique columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    types.SimpleNamespace
        SimpleNamespace with `unique` and `non_unique` column names indices.
    """

    mask_unique_columns = df.apply(unique_cols)

    columns = SimpleNamespace()
    columns.unique = df.columns[mask_unique_columns]
    columns.non_unique = df.columns[~mask_unique_columns]
    return columns


def prop_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.index.value_counts()
    prop = (counts > 1).sum() / len(counts)
    return 1 - prop


def replace_with(string_key: str, replace: str = "()/", replace_with: str = '') -> str:
    for symbol in replace:
        string_key = string_key.replace(symbol, replace_with)
    return string_key


def index_to_dict(index: pd.Index) -> dict:
    cols = {replace_with(col.replace(' ', '_').replace(
        '-', '_')): col for col in index}
    return cols


def get_columns_accessor(df: pd.DataFrame, all_lower_case=False) -> omegaconf.OmegaConf:
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("MultiIndex not supported.")
    cols = index_to_dict(df.columns)
    if all_lower_case:
        cols = {k.lower(): v for k, v in cols.items()}
    return omegaconf.OmegaConf.create(cols)


def get_columns_accessor_from_iterable(cols: Iterable[str],
                                       all_lower_case=False) -> omegaconf.OmegaConf:
    cols = index_to_dict(cols)
    if all_lower_case:
        cols = {k.lower(): v for k, v in cols.items()}
    return omegaconf.OmegaConf.create(cols)


def select_max_by(df: pd.DataFrame, grouping_columns: list, selection_column: str) -> pd.DataFrame:
    df = df.sort_values(by=[*grouping_columns, selection_column], ascending=False)
    df = df.drop_duplicates(subset=grouping_columns,
                            keep='first')
    return df


def get_columns_namedtuple(df: pd.DataFrame) -> namedtuple:
    """Create namedtuple instance of column names.
    Spaces in column names are replaced with underscores in the look-up.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame

    Returns
    -------
    namedtuple
        NamedTuple instance with columns as attributes.
    """
    columns = df.columns.to_list()
    column_keys = [x.replace(' ', '_') for x in columns]
    ColumnsNamedTuple = namedtuple('Columns', column_keys)
    return ColumnsNamedTuple(**{k: v for k, v in zip(column_keys, columns)})


def highlight_min(s: pd.Series) -> list:
    """Highlight the min in a Series yellow for using in pandas.DataFrame.style

    Parameters
    ----------
    s : pd.Series
        Pandas Series

    Returns
    -------
    list
        list of strings containing the background color for the values speciefied.
        To be used as `pandas.DataFrame.style.apply(highlight_min)`
    """
    to_highlight = s == s.min()
    return ['background-color: yellow' if v else '' for v in to_highlight]


def _add_indices(array: np.array, original_df: pd.DataFrame,
                 index_only: bool = False) -> pd.DataFrame:
    index = original_df.index
    columns = None
    if not index_only:
        columns = original_df.columns
    return pd.DataFrame(array, index=index, columns=columns)


def interpolate(wide_df: pd.DataFrame, name='interpolated') -> pd.DataFrame:
    """Interpolate NA values with the values before and after.
    Uses n=3 replicates.
    First rows replicates are the two following.
    Last rows replicates are the two preceding.

    Parameters
    ----------
    wide_df : pd.DataFrame
        rows are sample, columns are measurements
    name : str, optional
        name for measurement in columns, by default 'replicates'

    Returns
    -------
    pd.DataFrame
        pd.DataFrame in long-format
    """
    mask = wide_df.isna()
    first_row = wide_df.iloc[0].copy()
    last_row = wide_df.iloc[-1].copy()

    m = first_row.isna()
    first_row.loc[m] = wide_df.iloc[1:3, m.to_list()].mean()

    m = last_row.isna()
    last_row.loc[m] = wide_df.iloc[-3:-1, m.to_list()].mean()

    ret = wide_df.interpolate(
        method='linear', limit_direction='both', limit=1, axis=0)
    ret.iloc[0] = first_row
    ret.iloc[-1] = last_row

    ret = ret[mask].stack().dropna().squeeze()  # does not work with MultiIndex columns
    ret.rename(name, inplace=True)
    return ret


def flatten_dict_of_dicts(d: dict, parent_key: str = '') -> dict:
    """Build tuples for nested dictionaries for use as `pandas.MultiIndex`.

    Parameters
    ----------
    d : dict
        Nested dictionary for which all keys are flattened to tuples.
    parent_key : str, optional
        Outer key (used for recursion), by default ''

    Returns
    -------
    dict
        Flattend dictionary with tuple keys: {(outer_key, ..., inner_key) : value}
    """
    # simplified and adapted from: https://stackoverflow.com/a/6027615/9684872
    items = []
    for k, v in d.items():
        new_key = parent_key + (k,) if parent_key else (k,)
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict_of_dicts(v, parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def key_map(d: dict) -> dict:
    """Build a schema of dicts

    Parameters
    ----------
    d : dict
        dictionary of dictionaries

    Returns
    -------
    dict
        Key map of dictionaries
    """
    ret = {}
    _keys = ()
    for k, v in d.items():
        if isinstance(v, dict):
            ret[k] = key_map(v)
        else:
            _keys = (_keys) + (k, )
    if _keys:
        if ret:
            print(
                f"Dictionaries are not of the same length: {_keys = } and {ret = }")
            for k in _keys:
                ret[k] = None
        else:
            return _keys
    return ret


printable = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '


def parse_query_expression(s: str, printable: str = printable) -> str:
    """Parse a query expression for pd.DataFrame.query to a file name.
    Removes all characters not listed in printable."""
    return ''.join(filter(lambda x: x in printable, s))


def length(x):
    """Len function which return 0 if object (probably np.nan) has no length.
    Otherwise return length of list, pandas.Series, numpy.array, dict, etc."""
    try:
        return len(x)
    except BaseException:
        return 0


def get_last_index_matching_proportion(df_counts: pd.DataFrame,
                                       prop: float = 0.25,
                                       prop_col: str = 'proportion') -> object:
    """df_counts needs to be sorted by "prop_col" (descending).

    Parameters
    ----------
    df_counts : pd.DataFrame
        df counts with ascending values along proportion column.
        Index should be unique.
    prop : float, optional
        cutoff, inclusive, by default 0.25
    prop_col : str, optional
        column name for proportion, by default 'proportion'

    Returns
    -------
    object
        Index value for cutoff
    """
    assert df_counts.index.is_unique
    mask = df_counts[prop_col] >= prop
    idx_cutoff = df_counts[prop_col].loc[mask].tail(1).index[0]
    return idx_cutoff


def get_lower_whiskers(df: pd.DataFrame, factor: float = 1.5) -> pd.Series:
    ret = df.describe()
    iqr = ret.loc['75%'] - ret.loc['25%']
    ret = ret.loc['25%'] - iqr * factor
    return ret


def get_counts_per_bin(df: pd.DataFrame,
                       bins: range,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Return counts per bin for selected columns in DataFrame."""
    counts_per_bin = dict()
    if columns is None:
        columns = df.columns.to_list()
    for col in columns:
        _series = (pd.cut(df[col], bins=bins).to_frame().groupby(col).size())
        _series.index.name = 'bin'
        counts_per_bin[col] = _series
    counts_per_bin = pd.DataFrame(counts_per_bin)
    return counts_per_bin
