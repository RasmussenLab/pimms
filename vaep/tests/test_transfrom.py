import pandas as pd
import numpy as np
import numpy.testing as npt
from sklearn import preprocessing

from vaep.transform import log
from vaep.transform import StandardScaler, ShiftedStandardScaler


def test_log():
    row = pd.Series([np.NaN, 0.0, np.exp(1), np.exp(2)])
    row = log(row)
    assert row.equals(pd.Series([np.NaN, np.NaN, 1.0, 2.0]))


def test_StandardScaler():
    X = pd.DataFrame(np.array([[2, None], [3, 2], [4, 6]]))
    npt.assert_almost_equal(
        preprocessing.StandardScaler().fit(X).transform(X),
        StandardScaler().fit(X).transform(X).to_numpy()
    )


def test_ShiftedStandardScaler():
    X = np.random.random(size=(50, 10))
    scaler = ShiftedStandardScaler().fit(X)
    X_new = scaler.transform(X, copy=True)
    X_new = scaler.inverse_transform(X_new)
    npt.assert_almost_equal(X_new, X)
