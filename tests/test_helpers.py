import numpy as np

from vaep.utils import create_random_missing_data


def test_create_random_missing_data():
    data = create_random_missing_data(N=43, M=13, prop_missing=0.2)
    assert data.shape == (43, 13)
    assert np.isnan(data).sum()
    assert abs((float(np.isnan(data).sum()) / (43 * 13)) - 0.2) < 0.1
