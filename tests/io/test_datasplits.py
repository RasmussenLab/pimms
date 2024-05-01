import numpy as np
import pandas as pd
from vaep.io.datasplits import DataSplits, wide_format
import pytest
import numpy.testing as npt

N, M = 10, 4

X = np.random.rand(N, M)
df = (pd.DataFrame(X,
                   index=[f'sample_{i}' for i in range(N)],
                   columns=(f'feat_{i}' for i in range(M)))
      .rename_axis('Sample ID')
      .rename_axis('Feature Name', axis=1))

_splits = {'train_X': df.iloc[:int(N * 0.6)],
           'val_y': df.iloc[int(N * 0.6):int(N * 0.8)],
           'test_y': df.iloc[int(N * 0.8):]}


def test_DataSplits_iter():
    # expected = [(k, None) for k in list(DataSplits.__annotations__)]
    expected = [(k, None) for k in list(_splits)]
    splits = DataSplits(is_wide_format=None)
    assert sorted(list(splits)) == sorted(expected)


def test_DataSplits_dir():
    actual = sorted(dir(DataSplits(is_wide_format=False)))
    # expected = sorted(list(_splits))
    expected = ['dump', 'from_folder', 'interpolate', 'load', 'test_X', 'test_y',
                'to_long_format', 'to_wide_format', 'train_X', 'val_X', 'val_y']
    assert actual == expected


def test_load_missing_dir():
    splits = DataSplits(is_wide_format=False)
    with pytest.raises(AssertionError):
        splits.load(folder='non_exisiting')


def test_dump_empty(tmp_path):
    splits = DataSplits(is_wide_format=False)
    with pytest.raises(ValueError):
        splits.dump(tmp_path)


def test_dump_load(tmp_path):
    splits = DataSplits(**_splits, is_wide_format=True)
    splits.dump(folder=tmp_path)
    splits.load(folder=tmp_path)
    splits = DataSplits.from_folder(folder=tmp_path)

    splits = DataSplits(is_wide_format=None)
    splits.load(folder=tmp_path, use_wide_format=True)
    assert splits.train_X is not _splits['train_X']

    npt.assert_almost_equal(_splits['train_X'].values, splits.train_X)
    # #ToDo: Index and Column names are not yet correctly set
    # assert splits.train_X.equals(_splits['train_X'])


def test_to_long_format(tmp_path):
    splits = DataSplits(**_splits, is_wide_format=True)
    splits.dump(folder=tmp_path)
    splits = DataSplits(is_wide_format=None)
    splits.load(folder=tmp_path, use_wide_format=True)
    assert splits._is_wide
    expected = splits.val_y.copy()
    splits.to_long_format()
    assert not splits._is_wide
    splits.to_wide_format()
    assert splits.val_y is not expected
    assert splits.val_y.equals(expected)


def test_to_wide_format(tmp_path):
    splits = DataSplits(**_splits, is_wide_format=True)
    splits.dump(folder=tmp_path)
    splits = DataSplits(is_wide_format=None)
    splits.load(folder=tmp_path, use_wide_format=False)
    assert not splits._is_wide
    expected = splits.val_y.copy()
    splits.to_wide_format()
    assert splits._is_wide
    splits.to_long_format()
    assert splits.val_y is not expected
    assert splits.val_y.equals(expected)


def test_interpolate():
    splits = DataSplits(**_splits, is_wide_format=True)
    splits._is_wide = True  # ToDo. Is not correctly set when init is called.
    with pytest.raises(AttributeError):
        _ = splits.interpolate('non-existing')

    _ = splits.interpolate('train_X')

    with pytest.raises(AttributeError):
        _ = splits.interpolate('val_X')
    with pytest.raises(TypeError):
        _ = splits.interpolate(4)
