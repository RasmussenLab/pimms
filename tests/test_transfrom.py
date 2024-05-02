import numpy.testing as npt
import pandas as pd
import pytest
import sklearn
from sklearn import impute, preprocessing

from vaep.io.datasets import to_tensor
from vaep.transform import VaepPipeline


def test_Vaep_Pipeline():
    dae_default_pipeline = sklearn.pipeline.Pipeline(
        [
            ('normalize', preprocessing.StandardScaler()),
            ('impute', impute.SimpleImputer(add_indicator=False))  # True won't work
        ]
    )
    from random_data import data
    df = pd.DataFrame(data)
    mask = df.notna()
    # new procs, transform equal encode, inverse_transform equals decode
    dae_transforms = VaepPipeline(df, encode=dae_default_pipeline)
    res = dae_transforms.transform(df)
    assert isinstance(res, pd.DataFrame)
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
