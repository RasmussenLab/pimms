import collections.abc
from collections import namedtuple
from types import SimpleNamespace
import pandas as pd


def combine_value_counts(X: pd.DataFrame, dropna=True):
    """Pass a selection of columns to combine it's value counts.

    This performs no checks. Make sure the scale of the variables
    you pass is comparable.

    Parameters
    ----------
    X : pandas.DataFrame
        A DataFrame of several columns with values in a similar range.
    dropna : bool, optional
        Exclude NA values from counting, by default True

    Returns
    -------
    pandas.DataFrame
        DataFrame of combined value counts.
    """
    """
    """
    _df = pd.DataFrame()
    for col in X.columns:
        _df = _df.join(X[col].value_counts(dropna=dropna), how='outer')
    freq_targets = _df.sort_index()
    return freq_targets


def unique_cols(s: pd.Series):
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


def get_unique_non_unique_columns(df: pd.DataFrame):
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


def get_columns_namedtuple(df: pd.DataFrame):
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


def highlight_min(s):
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


def _add_indices(array, original_df, index_only=False):
    index = original_df.index
    columns = None
    if not index_only:
        columns = original_df.columns
    return pd.DataFrame(array, index=index, columns=columns)


def interpolate(wide_df: pd.DataFrame, name='replicates'):
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

    ret = ret[mask].stack().dropna()
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
    items = []
    for k, v in d.items():
        new_key = parent_key + (k,) if parent_key else (k,)
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict_of_dicts(v, parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)
