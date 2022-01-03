from pathlib import Path

import pandas as pd
import pytest

from vaep.io.datasplits import to_long_format
from vaep.sampling import feature_frequency, frequency_by_index

from vaep.utils import create_random_df


@pytest.fixture
def random_data():
    """Fixture to load random data."""
    return create_random_df(100, 15, prop_na=0.1)


@pytest.fixture
def example_data():
    """
    Fixture to load example data from a csv file for testing.
    """
    example_data_path = Path(__file__).resolve().parent / 'test_data.csv'
    return pd.read_csv(example_data_path, index_col='id')


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
