import helpers
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import torch
from numpy import nan

from vaep.io import datasets
from vaep.io.datasets import DatasetWithMaskAndNoTarget, DatasetWithTarget

data = np.random.random(size=(10, 5))
threshold = max(0.15, data.min() + 0.02)
mask = ~(data < threshold)
data_w_na = np.where(mask, data, np.nan)

assert (data != data_w_na).any()
assert (~np.isnan(data_w_na) == mask).all()


def test_PeptideDatasetInMemory_wo_Mask():
    train_ds = datasets.PeptideDatasetInMemory(data_w_na, fill_na=0.0)
    mask_isna = np.isnan(data_w_na)
    npt.assert_array_equal(train_ds.mask, mask_isna)
    npt.assert_almost_equal(data_w_na, train_ds.y)
    npt.assert_array_equal(
        train_ds.peptides[~train_ds.mask], train_ds.y[~train_ds.mask]
    )
    npt.assert_array_equal(train_ds.peptides == 0.0, ~mask)
    npt.assert_array_equal(train_ds.peptides == 0.0, train_ds.mask)


def test_DatasetWithMaskAndNoTarget():

    with pytest.raises(ValueError):
        DatasetWithMaskAndNoTarget(df=np.random.rand(10, 5))

    data = helpers.create_DataFrame()
    ds = DatasetWithMaskAndNoTarget(df=data)
    assert all(ds[-1][1] == torch.tensor([95, 96, 97, 98, 99], dtype=torch.int32))
    assert all(ds[-1][0] == torch.tensor([False, False, False, False, False]))


def test_DatasetWithTarget():
    data = helpers.create_DataFrame()
    ds = DatasetWithTarget(df=data)
    assert all(ds[-1][1] == torch.tensor([95, 96, 97, 98, 99], dtype=torch.int32))
    assert all(ds[-1][1] == torch.tensor([95, 96, 97, 98, 99], dtype=torch.int32))
    assert all(ds[-1][2] == torch.tensor([95, 96, 97, 98, 99], dtype=torch.int32))


def test_DatasetWithTargetSpecifyTarget():
    data = helpers.create_DataFrame()
    targets = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
    N, M = targets.shape
    for i in range(N):
        targets.iloc[i, i % M] = data.iloc[i, i % M]
        data.iloc[i, i % M] = np.nan

    ds = datasets.DatasetWithTargetSpecifyTarget(df=data, targets=targets)

    assert all(
        ds[-1][0] == torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    )
    torch.testing.assert_close(
        ds[-1][1],
        torch.tensor([95.0, 96.0, 97.0, 98.0, nan], dtype=torch.float32),
        equal_nan=True,
    )
    assert all(
        ds[-1][2] == torch.tensor([95.0, 96.0, 97.0, 98.0, 99.0], dtype=torch.float32)
    )


def get_example_data():
    targets = {
        6: {
            0: 26.220900810938048,
            1: 26.99154281369332,
            2: nan,
            3: nan,
            4: 28.431872162277948,
            5: 24.373006831943737,
            6: nan,
            7: nan,
            8: nan,
            9: nan,
            10: nan,
            11: nan,
            12: 27.181944902408738,
            13: 25.714911650080836,
            14: nan,
        },
        4: {
            0: nan,
            1: nan,
            2: 25.9033097961372,
            3: 29.61392051687617,
            4: nan,
            5: nan,
            6: nan,
            7: nan,
            8: 22.672194745133762,
            9: 25.04286715572299,
            10: nan,
            11: nan,
            12: nan,
            13: nan,
            14: 27.007771607333062,
        },
        0: {
            0: nan,
            1: nan,
            2: nan,
            3: nan,
            4: nan,
            5: nan,
            6: 27.644000604067763,
            7: nan,
            8: nan,
            9: nan,
            10: 23.1961478750206,
            11: nan,
            12: nan,
            13: nan,
            14: nan,
        },
        1: {
            0: nan,
            1: nan,
            2: nan,
            3: nan,
            4: nan,
            5: nan,
            6: nan,
            7: 26.80379574751157,
            8: nan,
            9: nan,
            10: nan,
            11: nan,
            12: nan,
            13: nan,
            14: nan,
        },
        5: {
            0: nan,
            1: nan,
            2: nan,
            3: nan,
            4: nan,
            5: nan,
            6: nan,
            7: nan,
            8: nan,
            9: nan,
            10: nan,
            11: 24.23674609840043,
            12: nan,
            13: nan,
            14: nan,
        },
    }

    data = {
        0: {
            0: 23.71432923730573,
            1: 24.917845078617184,
            2: 23.31366903064276,
            3: 25.92083503730988,
            4: 25.409860187753736,
            5: nan,
            6: nan,
            7: 20.993230967841804,
            8: 26.44436968540382,
            9: 27.67808101094618,
            10: nan,
            11: 27.267946461624568,
            12: nan,
            13: 21.205767395881473,
            14: 23.289197954749703,
        },
        1: {
            0: 23.009120461092184,
            1: 25.468933308196725,
            2: 25.007703920302433,
            3: 25.454818109164382,
            4: 25.562251032076684,
            5: 27.800964985425356,
            6: 25.46401844268311,
            7: nan,
            8: 21.06435859443234,
            9: 26.53528962285722,
            10: 28.334273071386868,
            11: 26.21361598812065,
            12: 26.016338863149223,
            13: 22.264927781719752,
            14: 24.571585004609357,
        },
        2: {
            0: 25.322652614961317,
            1: 22.915995048122916,
            2: nan,
            3: nan,
            4: 24.290967310011215,
            5: 23.447554991686353,
            6: nan,
            7: 26.107959643816418,
            8: 23.253808429522262,
            9: nan,
            10: 24.374533956685088,
            11: 23.989901338279623,
            12: 24.500020668131395,
            13: 26.389842757424784,
            14: nan,
        },
        3: {
            0: 27.535394665679163,
            1: 24.15354899711814,
            2: 25.854438476227674,
            3: nan,
            4: 22.521487231136202,
            5: 28.289066709807237,
            6: 23.880579227014348,
            7: nan,
            8: 27.777054696818883,
            9: 22.980933816432707,
            10: 28.323095228640135,
            11: nan,
            12: 26.57374954821719,
            13: 23.766966814678767,
            14: nan,
        },
        4: {
            0: 22.35907959911305,
            1: nan,
            2: nan,
            3: nan,
            4: 26.927564413252995,
            5: 23.953910163147384,
            6: 24.874682047196444,
            7: 25.150676295440704,
            8: nan,
            9: nan,
            10: 23.74293493785907,
            11: 25.126081245920954,
            12: nan,
            13: 25.72971061050739,
            14: nan,
        },
        5: {
            0: 27.76908766045489,
            1: 23.097347607619813,
            2: 22.80553197803914,
            3: 21.86796097887063,
            4: 26.33547904887626,
            5: 27.88885308507015,
            6: nan,
            7: 26.31494838890394,
            8: 23.527389982569765,
            9: 24.372064526940033,
            10: 25.161065651989915,
            11: nan,
            12: 24.835271805650216,
            13: 25.2322376364396,
            14: 23.624665336496243,
        },
        6: {
            0: nan,
            1: nan,
            2: 26.084714169356463,
            3: 23.186037795469158,
            4: nan,
            5: nan,
            6: nan,
            7: 23.760299986897984,
            8: 21.122235545588637,
            9: nan,
            10: 23.409893505677946,
            11: 23.002282856949503,
            12: nan,
            13: nan,
            14: 22.443571469776245,
        },
    }

    return data, targets


def test_DatasetWithTargetSpecifyTarget_floats():
    data, targets = get_example_data()
    data = pd.DataFrame(data)
    targets = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
    N, M = targets.shape
    for i in range(N):
        targets.iloc[i, i % M] = data.iloc[i, i % M]
        data.iloc[i, i % M] = np.nan

    ds = datasets.DatasetWithTargetSpecifyTarget(df=data, targets=targets)

    torch.testing.assert_close(
        ds[-1][0], torch.tensor([0., 1., 1., 1.,
                                 1., 1., 1.], dtype=torch.float32)
    )
    torch.testing.assert_close(
        ds[-1][1],
        torch.tensor([nan, 24.5716, nan, nan,
                      nan, 23.6247, 22.4436], dtype=torch.float32),
        equal_nan=True)
    torch.testing.assert_close(
        ds[-1][2],
        torch.tensor([23.2892, 24.5716, nan,
                      nan, nan, 23.6247, 22.4436], dtype=torch.float32),
        equal_nan=True)
