from pathlib import Path

import pandas as pd
import pytest

from vaep.io.datasplits import to_long_format
from vaep.sampling import feature_frequency, frequency_by_index, sample_data

from vaep.utils import create_random_df


@pytest.fixture
def random_data():
    """Fixture to load random data."""
    return create_random_df(100, 10, prop_na=0.1).rename_axis('Sample ID').rename_axis('feat name', axis=1)


@pytest.fixture
def example_data():
    """
    Fixture to load example data from a csv file for testing.
    """
    example_data_path = Path(__file__).resolve().parent / 'test_data.csv'
    return pd.read_csv(example_data_path, index_col='id').rename_axis('Sample ID').rename_axis('feat name', axis=1)


def test_feature_frequency(random_data):
    X = random_data
    assert all(feature_frequency(X)
               ==
               frequency_by_index(to_long_format(X),
                                  sample_index_to_drop='Sample ID'))


def test_frequency_by_index(example_data):
    X = example_data
    assert all(feature_frequency(X)
               ==
               frequency_by_index(to_long_format(X),
                                  sample_index_to_drop=0))


def test_sample_data(random_data):
    X = random_data
    freq = feature_frequency(X)
    excluded_feat = freq.sample(2).index.to_list()
    freq.loc[excluded_feat] = 0
    X = to_long_format(X).squeeze()
    # ValueError: Fewer non-zero entries in p than size -> too many feat set to zero
    series_sampled, series_not_sampled = sample_data(
        X, 0, frac=0.70, weights=freq)
    assert len(X) == len(
        series_sampled) + len(series_not_sampled)
    assert X.index.difference(
        series_sampled.index.append(series_not_sampled.index)).empty
    idx_excluded = series_sampled.index.isin(excluded_feat, level=1)
    assert series_sampled.loc[idx_excluded].empty
    idx_excluded = series_not_sampled.index.isin(excluded_feat, level=1)
    assert not series_not_sampled.loc[idx_excluded].empty
