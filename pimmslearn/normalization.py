import logging

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_by_median(df_wide: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
    """Normalize by median. Level using global median of medians.

    Parameters
    ----------
    df_wide : pd.DataFrame
        DataFrame with samples as rows and features as columns
    axis : int, optional
        Axis to normalize, by default 1 (i.e. by row/sample)

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame
    """
    medians = df_wide.median(axis=axis)
    global_median = medians.median()
    df_wide = df_wide.subtract(medians, axis=1 - axis) + global_median
    return df_wide


def normalize_sceptre(quant: pd.DataFrame,
                      iter_thresh: float = 1.1,
                      iter_max: int = 10,
                      check_convex=True) -> pd.DataFrame:
    """Normalize by sample and channel as in SCeptre paper. Code adapted to work
    with current pandas versions.

    Parameters
    ----------
    quant : pd.DataFrame
        MulitIndex columns with two levels: File ID and Channel
        Not log transformed.
    iter_thresh : float, optional
        treshold for maximum absolute deviation in iteration, by default 1.1
    iter_max : int, optional
        maximum number of iterations to check for convergence, by default 10

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with same index and columns as input
    """
    max_dev_old = None
    for i in range(iter_max):  # iterate to converge to normalized channel and file
        quant_0 = quant.copy()

        # file bias normalization
        # calculate median for each protein in each sample
        med = quant.groupby(axis=0, level=0).median()
        # calculate the factors needed for a median shift
        med_tot = med.median(axis=0)
        factors = med.divide(med_tot, axis=1)
        quant = quant.divide(factors)

        # channel bias normalization
        # calculate median for each protein in each channel
        med = quant.groupby(axis=0, level=1).median()
        # calculate the factors needed for a median shift
        med_tot = med.median(axis=1)
        factors = med.divide(med_tot, axis=0)
        quant = quant.divide(factors)
        # stop iterating when the change in quant to the previous iteration is below iter_thresh
        max_dev = abs(quant - quant_0).max().max()
        median_dev = abs(quant - quant_0).median().median()
        print(f"Max deviation: {max_dev:.2f}, median deviation: {median_dev:.2f}")
        if (median_dev) <= iter_thresh:
            print(f"Max deviation: {max_dev:.2f}")
            print(f"Median deviation: {median_dev:.2f}")
            break
        if i > 0 and check_convex and max_dev_old < max_dev:
            raise ValueError("Non-convex behaviour. old max deviation smaller than current.")
        print("performed {} iterations, max-dev: {:.2f}".format(i + 1, max_dev))
        max_dev_old = max_dev
    return quant
