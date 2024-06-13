import logging

import pandas as pd

logger = logging.getLogger(__name__)


def select_features(df: pd.DataFrame,
                    feat_prevalence: float = .2,
                    axis: int = 0) -> pd.DataFrame:
    """Select features or samples with a minimum prevalence.
    """
    N = df.shape[axis]
    minimum_freq = N * feat_prevalence
    freq = df.notna().sum(axis=axis)
    mask = freq >= minimum_freq
    axis_synonym = "index" if axis == 0 else "columns"
    logger.info(f"Drop {(~mask).sum()} along axis {axis} ({axis_synonym}).")
    freq = freq.loc[mask]
    if axis == 0:
        df = df.loc[:, mask]
    else:
        df = df.loc[mask]
    return df
