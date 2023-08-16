"""
Functionality to handle protein and peptide datasets.
"""
import numpy as np
import pandas as pd

# coverage


def coverage(X: pd.DataFrame, coverage_col: float, coverage_row: float):
    """Select proteins by column depending on their coverage.
    Of these selected proteins, where the rows have a certain number of overall proteins.
    """
    mask_col = X.isnull().mean() <= 1 - coverage_col
    df = X.loc[:, mask_col]
    mask_row = df.isnull().mean(axis=1) <= 1 - coverage_row
    df = df.loc[mask_row, :]
    return df


def compute_stats_missing(X: pd.DataFrame,
                          col_no_missing: str = 'no_missing',
                          col_no_identified: str = 'no_identified',
                          col_prop_samples: str = 'prop_samples') -> pd.DataFrame:
    """Dataset of repeated samples indicating if an observation
    has the variables observed or missing x in {0,1}"""
    if X.index.name:
        index_col = X.index.name
    else:
        index_col = 'INDEX'
    sample_stats = X.index.to_frame(index=False).reset_index()
    sample_stats.columns = ['SampleID_int', index_col]
    sample_stats.set_index(index_col, inplace=True)

    sample_stats[col_no_identified] = X.sum(axis=1)
    sample_stats[col_no_missing] = (X == 0).sum(axis=1)

    assert all(sample_stats[[col_no_identified, col_no_missing]].sum(
        axis=1) == X.shape[1])
    sample_stats = sample_stats.sort_values(
        by=col_no_identified, ascending=False)
    sample_stats[col_prop_samples] = np.array(
        range(1, len(sample_stats) + 1)) / len(sample_stats)
    return sample_stats


def get_sorted_not_missing(X: pd.DataFrame) -> pd.DataFrame:
    """Return a Dataframe with missing values. Order columns by degree of completness
    over columns from variables least to most shared among observations."""
    X = X.notna().astype(int)
    return X[X.mean().sort_values().index]
