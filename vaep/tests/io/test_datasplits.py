import numpy as np
import pandas as pd
from vaep.io.datasplits import DataSplits
import pytest

N, M = 10, 4

X = np.random.rand(N, M)
df = pd.DataFrame(X)

y = np.random.random(N)
y = pd.Series(y)

_splits = {'train_X': df.iloc[:int(N*0.6)],
           'val_X': df.iloc[int(N*0.6):int(N*0.8)],
           'val_y': y.iloc[int(N*0.6):int(N*0.8)],
           'test_X': df.iloc[int(N*0.8):],
           'test_y': y.iloc[int(N*0.8):]}


def test_DataSplits_iter():
    # expected = [(k, None) for k in list(DataSplits.__annotations__)]
    expected = [(k, None) for k in list(_splits)]
    splits = DataSplits()
    assert sorted(list(splits)) == sorted(expected)


def test_DataSplits_dir():
    actual = sorted(dir(DataSplits()))
    expected = sorted(list(_splits))
    assert actual == expected


def test_load_missing_dir():
    splits = DataSplits()
    with pytest.raises(AssertionError):
        splits.load(folder='non_exisiting')


def test_dump_empty(tmp_path):
    splits = DataSplits()
    with pytest.raises(ValueError):
        splits.dump(tmp_path)


def test_dump_load(tmp_path):
    splits = DataSplits(**_splits)
    splits.dump(folder=tmp_path)
    splits.load(folder=tmp_path)
    splits = DataSplits.from_folder(folder=tmp_path)
