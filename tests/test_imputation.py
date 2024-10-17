"""
# Test Data set was created from a sample by shuffling:

fraction_missing = proteins.notna().mean()

data = data[data.columns[fraction_missing > 0.4]]
N_FEAT = 200
N_FEAT_digits = len(str(N_FEAT))
data = data.sample(N_FEAT, axis=1)
data.columns = [f"P{i:0{N_FEAT_digits}d}" for i in range(N_FEAT)]
data.reset_index(inplace=True)
data.drop('index', axis=1, inplace=True)
data.apply(numpy.random.shuffle, axis=1)
data.to_csv('test_data.csv')
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pimmslearn.imputation import impute_shifted_normal


@pytest.fixture
def example_data():
    """
    Fixture to load example data from a csv file for testing.
    """
    example_data_path = Path(__file__).resolve().parent / 'test_data.csv'
    return pd.read_csv(example_data_path, index_col='id')


@pytest.mark.parametrize('axis', [0, 1])
def test_impute_shifted_normal(example_data, axis):
    mean_shift = 1.8
    # remove zeros as these lead to -inf
    example_data = np.log2(example_data.replace({0.0: np.nan})
                           ).dropna(thresh=10, axis=1 - axis)
    N, M = example_data.shape
    mask_observed = example_data.notna()
    imputed = impute_shifted_normal(example_data, axis=axis, mean_shift=mean_shift)
    assert len(imputed) == ((N * M) - len(example_data.stack()))

    if axis == 1:
        min_N = int(len(example_data) * 0.6)
        selected = example_data.dropna(axis=1, thresh=min_N)
    elif axis == 0:
        min_M = int(example_data.shape[1] * 0.6)
        selected = example_data.dropna(axis=0, thresh=min_M)

    mean = selected.mean(axis=axis)
    std = selected.std(axis=axis)
    mean_shifted = mean - (std * mean_shift)

    mean_imputed = imputed.unstack().mean(axis=axis)
    assert (mean_shifted - mean_imputed).abs().max() < 0.35
