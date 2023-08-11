import pytest
import pandas as pd
import numpy as np
import numpy.testing as npt



import sklearn
from sklearn import preprocessing
from sklearn import impute

from vaep.transform import log
from vaep.transform import StandardScaler, ShiftedStandardScaler, VaepPipeline
from vaep.io.datasets import to_tensor

# not used anywhere
# def test_log():
#     row = pd.Series([np.NaN, 0.0, np.exp(1), np.exp(2)])
#     row = log(row)
#     assert row.equals(pd.Series([np.NaN, np.NaN, 1.0, 2.0]))


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


# from sklearn import preprocessing


def test_Vaep_Pipeline():
    dae_default_pipeline = sklearn.pipeline.Pipeline(
        [
            ('normalize', preprocessing.StandardScaler()),
            ('impute', impute.SimpleImputer(add_indicator=False)) # True won't work
        ]
    )
    from random_data import data
    df = pd.DataFrame(data)
    mask = df.notna()
    # new procs, transform equal encode, inverse_transform equals decode
    dae_transforms = VaepPipeline(df, encode=dae_default_pipeline)
    res = dae_transforms.transform(df)
    assert type(res)  == pd.DataFrame
    with pytest.raises(ValueError):
        res = dae_transforms.inverse_transform(res)  # pd.DataFrame
    with pytest.raises(ValueError):    
        _ = dae_transforms.inverse_transform(res.iloc[0])  # pd.DataFrame
    with pytest.raises(ValueError):    
        _ = dae_transforms.inverse_transform(res.loc['sample_156'])  # pd.DataFrame
    with pytest.raises(ValueError):    
        _ = dae_transforms.inverse_transform(to_tensor(res))  # torch.Tensor
    with pytest.raises(ValueError):    
        _ = dae_transforms.inverse_transform(res.values)  # numpy.array
    with pytest.raises(ValueError):    
        _ = dae_transforms.inverse_transform(res.values[0])  # single sample
    dae_transforms = VaepPipeline(df, encode=dae_default_pipeline, decode=['normalize'])
    res = dae_transforms.transform(df)
    res = dae_transforms.inverse_transform(res)
    npt.assert_array_almost_equal(df.values[mask], res.values[mask])