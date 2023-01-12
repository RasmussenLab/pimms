from typing import Union, Tuple

import pandas as pd


def feature_frequency(df_wide: pd.DataFrame, measure_name: str = 'freq') -> pd.Series:
    """Generate frequency table based on singly indexed (both axes) DataFrame.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Singly indexed DataFrame with singly indexed columns (no MultiIndex)
    measure_name : str, optional
        Name of the returned series, by default 'freq'

    Returns
    -------
    pd.Series
        Frequency on non-missing entries per feature (column). 
    """
    # if hasattr(df_wide.columns, "levels"): # is columns.names always set?
    # is listed as attribute: https://pandas.pydata.org/docs/reference/api/pandas.Index.html
    _df_feat = df_wide.stack(df_wide.columns.names) # ensure that columns are named

    _df_feat = _df_feat.to_frame(measure_name)
    # implicit as stack puts column index in the last position (here: 1)
    _df_feat = _df_feat.reset_index(0, drop=True)
    level = list(range(len(_df_feat.index.names)))
    freq_per_feat = _df_feat.notna().groupby(level=level).sum()
    return freq_per_feat.squeeze()


def frequency_by_index(df_long: pd.DataFrame, sample_index_to_drop: Union[str, int]) -> pd.Series:
    """Generate frequency table based on an index level of a 2D multiindex.

    Parameters
    ----------
    df_long : pd.DataFrame
        One column, 2D multindexed DataFrame
    sample_index_to_drop : Union[str, int]
        index name or position not to use

    Returns
    -------
    pd.Series
        frequency of index categories in table (not missing)
    """
    # potentially more than one index
    # to_remove = tuple(set(df_long.index.names) - set([by_index]))
    _df_feat = df_long.reset_index(level=sample_index_to_drop, drop=True)
    # same as in feature_frequency
    freq_per_feat = _df_feat.notna().groupby(level=0, observed=True).sum()
    return freq_per_feat.squeeze()


def sample_data(series: pd.Series, sample_index_to_drop: Union[str, int],
                frac=0.95, weights: pd.Series = None,
                random_state=42) -> Tuple[pd.Series, pd.Series]:
    """sample from doubly indexed series with sample index and feature index.

    Parameters
    ----------
    series : pd.Series
        Long-format data in pd.Series. Index name is feature name. 2 dimensional 
        MultiIndex. 
    sample_index_to_drop : Union[str, int]
        Sample index (as str or integer Index position). Unit to group by (i.e. Samples)
    frac : float, optional
        Percentage of single unit (sample) to sample, by default 0.95
    weights : pd.Series, optional
        Weights to pass on for sampling on a single group, by default None
    random_state : int, optional
        Random state to use for sampling procedure, by default 42

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        First series contains the entries sampled, whereas the second series contains the
        entires not sampled from the orginally passed series.
    """
    index_names = series.index.names
    new_column = index_names[sample_index_to_drop]
    df = series.to_frame('intensity').reset_index(sample_index_to_drop)

    df_sampled = df.groupby(by=new_column).sample(
        frac=frac, weights=weights, random_state=random_state)
    series_sampled = df_sampled.reset_index().set_index(index_names).squeeze()

    idx_diff = series.index.difference(series_sampled.index)
    series_not_sampled = series.loc[idx_diff]
    return series_sampled, series_not_sampled
