import pandas as pd
import pytest

from vaep.imputation import * # ToDo: Choose function

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

data = pd.read_csv('test_data.csv', index_col='id')

# def test_impute_missing():
#     pass


def test_imputation_KNN(example_data):
    threshold = 0.55
    data = example_data
    data_transformed = imputation_KNN(data.copy(), threshold=threshold)
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

