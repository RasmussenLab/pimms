import numpy as np
import numpy.testing as npt

from vaep.io.datasets import PeptideDatasetInMemory

data = np.random.random(size=(10,5))
mask = ~(data < 0.1)
data_w_na = np.where(mask, data, np.nan)

assert (data != data_w_na).any()
assert (~np.isnan(data_w_na) == mask).all()

def test_PeptideDatasetInMemory_wo_Mask():
    train_ds = PeptideDatasetInMemory(data_w_na, fill_na=0.0)
    npt.assert_array_equal(train_ds.mask, mask)
    npt.assert_almost_equal(data[mask], train_ds.y[train_ds.mask])
    npt.assert_array_equal(train_ds.peptides[train_ds.mask], train_ds.y[train_ds.mask])
    npt.assert_array_equal(train_ds.peptides == 0.0, ~mask)


