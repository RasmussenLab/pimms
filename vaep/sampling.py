import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd

from vaep.io.datasplits import DataSplits

logger = logging.getLogger(__name__)


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
    _df_feat = df_wide.stack(df_wide.columns.names)  # ensure that columns are named

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


def sample_mnar_mcar(df_long: pd.DataFrame,
                     frac_non_train: float,
                     frac_mnar: float,
                     random_state: int = 42
                     ) -> Tuple[DataSplits, pd.Series, pd.Series, pd.Series]:
    """Sampling of data for MNAR/MCAR simulation. The function samples from the df_long
    DataFrame and returns the training, validation and test splits in dhte DataSplits object.


    Select features as described in
    > Lazar, Cosmin, Laurent Gatto, Myriam Ferro, Christophe Bruley, and Thomas Burger. 2016.
    > “Accounting for the Multiple Natures of Missing Values in Label-Free Quantitative
    > Proteomics Data Sets to Compare Imputation Strategies.”
    > Journal of Proteome Research 15 (4): 1116–25.

    - select MNAR based on threshold matrix on quantile
    - specify MNAR and MCAR proportions in validation and test set
    - use needed MNAR as specified by `frac_mnar`
    - sample MCAR from the remaining data
    - distribute MNAR and MCAR in validation and test set

    Parameters
    ----------
    df_long : pd.DataFrame
        intensities in long format with unique index.
    frac_non_train : float
        proprotion of data in df_long to be used for evaluation in total
        in validation and test split
    frac_mnar : float
        Frac of simulated data to be missing not at random (MNAR)
    random_state : int, optional
        random seed for reproducibility, by default 42

    Returns
    -------
    Tuple[DataSplits, pd.Series, pd.Series, pd.Series]
        datasplits, thresholds, fake_na_mcar, fake_na_mnar

        Containing training, validation and test splits, as well as the thresholds,
        mcar and mnar simulated missing intensities.
    """
    assert 0.0 <= frac_mnar <= 1.0, "Fraction must be between 0 and 1"

    thresholds = get_thresholds(df_long, frac_non_train, random_state)
    mask = df_long.squeeze() < thresholds
    N = len(df_long)
    logger.info(f"{int(N * frac_non_train) = :,d}")
    # Sample MNAR based on threshold matrix and desired share
    N_MNAR = int(frac_non_train * frac_mnar * N)
    fake_na_mnar = df_long.loc[mask]
    if len(fake_na_mnar) > N_MNAR:
        fake_na_mnar = fake_na_mnar.sample(N_MNAR,
                                           random_state=random_state)
    # select MCAR from remaining intensities
    splits = DataSplits(is_wide_format=False)
    splits.train_X = df_long.loc[
        df_long.index.difference(
            fake_na_mnar.index)
    ]
    logger.info(f"{len(fake_na_mnar) = :,d}")
    N_MCAR = int(N * (1 - frac_mnar) * frac_non_train)
    fake_na_mcar = splits.train_X.sample(N_MCAR,
                                         random_state=random_state)
    logger.info(f"{len(splits.train_X) = :,d}")

    fake_na = pd.concat([fake_na_mcar, fake_na_mnar]).squeeze()
    logger.info(f"{len(fake_na) = :,d}")

    logger.info(f"{len(fake_na_mcar) = :,d}")
    splits.train_X = (splits
                      .train_X
                      .loc[splits
                           .train_X
                           .index
                           .difference(
                               fake_na_mcar.index)]
                      ).squeeze()
    # Distribute MNAR and MCAR in validation and test set
    splits.val_y = fake_na.sample(frac=0.5, random_state=random_state)
    splits.test_y = fake_na.loc[fake_na.index.difference(splits.val_y.index)]

    assert len(fake_na) + len(splits.train_X) == len(df_long)
    return splits, thresholds, fake_na_mcar, fake_na_mnar


def get_thresholds(df_long: pd.DataFrame, frac_non_train: float,
                   random_state: int) -> pd.Series:
    """Get thresholds for MNAR/MCAR sampling. Thresholds are sampled from a normal
    distrubiton with a mean of the quantile of the simulated missing data.

    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format data in pd.DataFrame. Index name is feature name. 2 dimensional
        MultiIndex.
    frac_non_train : float
        Percentage of single unit (sample) to sample.
    random_state : int
        Random state to use for sampling procedure.

    Returns
    -------
    pd.Series
        Thresholds for MNAR/MCAR sampling.
    """
    quantile_frac = df_long.quantile(frac_non_train)
    rng = np.random.default_rng(random_state)
    thresholds = pd.Series(
        rng.normal(
            loc=float(quantile_frac),
            scale=float(0.3 * df_long.std()),
            size=len(df_long),
        ),
        index=df_long.index,
    )
    return thresholds


def check_split_integrity(splits: DataSplits) -> DataSplits:
    """Check if IDs in are only in validation or test data for rare cases.
    Returns the corrected splits."""
    diff = (splits
            .val_y
            .index
            .levels[-1]
            .difference(splits
                        .train_X
                        .index
                        .levels[-1]
                        ).to_list())
    if diff:
        logger.warning(f"Remove from val: {diff.to_list()}")
        to_remove = splits.val_y.loc[pd.IndexSlice[:, diff]]
        splits.train_X = pd.concat([splits.train_X, to_remove])
        splits.val_y = splits.val_y.drop(to_remove.index)

    diff = (splits
            .test_y
            .index
            .levels[-1]
            .difference(splits
                        .train_X
                        .index
                        .levels[-1]
                        ).to_list())
    if diff:
        logger.warning(f"Remove from test: {diff.to_list()}")
        to_remove = splits.test_y.loc[pd.IndexSlice[:, diff]]
        splits.train_X = pd.concat([splits.train_X, to_remove])
        splits.test_y = splits.test_y.drop(to_remove.index)
    return splits
