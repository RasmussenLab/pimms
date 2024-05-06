import numpy as np
import numpy.testing as npt
import pandas as pd

import vaep
from vaep.io.datasplits import DataSplits
from vaep.models import collab

N, M = 10, 4

X = np.random.rand(N, M)
df = pd.DataFrame(X,
                  index=[f'sample_{i}' for i in range(N)],
                  columns=(f'feat_{i}' for i in range(M)))

data = {'train_X': df.iloc[:int(N * 0.6)],
        'val_y': df.iloc[int(N * 0.6):int(N * 0.8)],
        'test_y': df.iloc[int(N * 0.8):]}

data = DataSplits(**data, is_wide_format=True)
assert data._is_wide
data.to_long_format()


def test_combine_data():
    N_train, N_val = len(data.train_X), len(data.val_y)
    X, frac = collab.combine_data(data.train_X, data.val_y)
    assert len(X) == N_train + N_val
    npt.assert_almost_equal(frac, N_val / (N_train + N_val))
