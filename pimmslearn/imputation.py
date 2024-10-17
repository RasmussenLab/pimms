"""
Reduce number of missing values of DDA massspectromety.

Imputation can be down by column.


"""
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


RANDOMSEED = 123


def impute_shifted_normal(df_wide: pd.DataFrame,
                          mean_shift: float = 1.8,
                          std_shrinkage: float = 0.3,
                          completeness: float = 0.6,
                          axis=1,
                          seed=RANDOMSEED) -> pd.Series:
    """Get replacements for missing values.

    Parameters
    ----------
    df_wide : pd.DataFrame
        DataFrame in wide format, contains missing
    mean_shift : float, optional
        shift mean of feature by factor of standard deviations, by default 1.8
    std_shrinkage : float, optional
        shrinks standard deviation by facotr, by default 0.3
    axis: int, optional
        axis along which to impute, by default 1 (i.e. mean and std per row)

    Returns
    -------
    pd.Series
        Series of imputed values in long format.
    """
    # add check if there ar e NaNs or inf in data? see tests
    # np.isinf(df_wide).values.sum()
    if axis == 1:
        min_N = int(len(df_wide) * completeness)
        selected = df_wide.dropna(axis=1, thresh=min_N)
    elif axis == 0:
        min_M = int(df_wide.shape[1] * completeness)
        selected = df_wide.dropna(axis=0, thresh=min_M)
    else:
        raise ValueError(
            "Please specify axis as 0 or 1, for axis along which to impute.")
    logger.info(
        f"Meand and standard deviation based on seleted data of shape {selected.shape}")
    mean = selected.mean(axis=axis)
    std = selected.std(axis=axis)
    mean_shifted = mean - (std * mean_shift)
    std_shrinked = std * std_shrinkage
    # rng=np.random.default_rng(seed=seed)
    # rng.normal()
    np.random.seed(seed)
    N, M = df_wide.shape
    if axis == 1:
        imputed_shifted_normal = pd.DataFrame(
            np.random.normal(mean_shifted, std_shrinked, size=(M, N)),
            index=df_wide.columns,
            columns=df_wide.index)
        imputed_shifted_normal = imputed_shifted_normal.T
    else:
        imputed_shifted_normal = pd.DataFrame(
            np.random.normal(mean_shifted, std_shrinked, size=(N, M)),
            index=df_wide.index,
            columns=df_wide.columns)
    imputed_shifted_normal = imputed_shifted_normal[df_wide.isna()].stack()
    return imputed_shifted_normal


def compute_moments_shift(observed: pd.Series, imputed: pd.Series,
                          names: Tuple[str, str] = ('observed', 'imputed')) -> Dict[str, float]:
    """Summary of overall shift of mean and std. dev. of predictions for a imputation method."""
    name_obs, name_model = names
    data = {name: {'mean': series.mean(), 'std': series.std()} for series, name in zip([observed, imputed], names)}
    observed, imputed = data[name_obs], data[name_model]
    data[name_model]['mean shift (in std)'] = (observed["mean"] - imputed["mean"]) / observed["std"]
    data[name_model]['std shrinkage'] = imputed["std"] / observed["std"]
    return data


def stats_by_level(series: pd.Series, index_level: int = 0, min_count: int = 5) -> pd.Series:
    """Count, mean and std. dev. by index level."""
    agg = series.groupby(level=index_level).agg(['count', 'mean', 'std'])
    agg = agg.loc[agg['count'] > min_count]
    return agg.mean()
