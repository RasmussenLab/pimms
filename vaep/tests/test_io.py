from pathlib import Path

import numpy as np
import numpy.testing as npt

import vaep.io 
from vaep.io.datasets import PeptideDatasetInMemory

data = np.random.random(size=(10,5))
mask = ~(data < 0.1)
data_w_na = np.where(mask, data, np.nan)

assert (data != data_w_na).any()
assert (~np.isnan(data_w_na) == mask).all()

def test_PeptideDatasetInMemory_wo_Mask():
    train_ds = PeptideDatasetInMemory(data_w_na, fill_na=0.0)
    mask_isna = np.isnan(data_w_na)
    npt.assert_array_equal(train_ds.mask, mask_isna)
    npt.assert_almost_equal(data_w_na, train_ds.y)
    npt.assert_array_equal(train_ds.peptides[~train_ds.mask], train_ds.y[~train_ds.mask])
    npt.assert_array_equal(train_ds.peptides == 0.0, ~mask)
    npt.assert_array_equal(train_ds.peptides == 0.0, train_ds.mask)


def test_relative_to():
    fpath = Path('project/runs/experiment_name/run')
    pwd  = 'project/runs/' # per defaut '.' (the current working directory)
    expected =  Path('experiment_name/run')
    acutal = vaep.io.resolve_path(fpath, pwd)
    assert expected == acutal

    # # no solution yet, expect chaning notebook pwd
    # fpath = Path('data/file')
    # # pwd is different subfolder
    # pwd  = 'root/home/project/runs/' # per defaut '.' (the current working directory)
    # expected =  Path('root/home/project/data/file')
    # acutal = vaep.io.resolve_path(fpath, pwd)
    # assert expected == acutal