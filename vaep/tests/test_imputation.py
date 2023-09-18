from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from vaep.imputation import imputation_KNN, imputation_normal_distribution, impute_shifted_normal
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


@pytest.fixture
def example_data():
    """
    Fixture to load example data from a csv file for testing.
    """
    example_data_path = Path(__file__).resolve().parent / 'test_data.csv'
    return pd.read_csv(example_data_path, index_col='id')

# def test_impute_missing():
#     pass


def test_imputation_KNN(example_data):
    threshold = 0.55
    data = example_data.copy()
    data_transformed = imputation_KNN(data, threshold=threshold)
    columns_to_impute = data.notnull().mean() >= threshold
    columns_to_impute = columns_to_impute[columns_to_impute].index
    assert all(data_transformed.loc[:, columns_to_impute].isna().sum() < 15)
    n_not_to_impute = data.loc[:,
                               data.notnull().mean() < threshold].isna().sum()
    assert all(data_transformed.loc[:, n_not_to_impute.index].isna().sum()
               == n_not_to_impute)


def test_imputation_normal_dist():
    log_intensities = pd.Series([26.0, np.nan, 24.0, 25.0, np.nan])
    imputed = imputation_normal_distribution(log_intensities)
    imputed = round(imputed, ndigits=5)
    assert imputed.equals(
        pd.Series([26.0, 22.87431, 24.0, 25.0, 22.87431])
    )

# def test_imputation_mixed_norm_KNN():
#     pass


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
