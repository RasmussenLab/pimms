from typing import Union

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
    _df_feat = df_wide.stack().to_frame(measure_name)
    # implicit as stack puts column index in the last position (here: 1)
    _df_feat = _df_feat.reset_index(0, drop=True)
    freq_per_feat = _df_feat.notna().groupby(level=0).sum()
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
    freq_per_feat = _df_feat.notna().groupby(level=0).sum()
    return freq_per_feat.squeeze()
