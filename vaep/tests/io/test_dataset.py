import numpy as np
import torch
import pytest

import helpers

from vaep.io.datasets import DatasetWithMaskAndNoTarget

def test_DatasetWithMaskAndNoTarget():    

    with pytest.raises(ValueError):
        DatasetWithMaskAndNoTarget(df=np.random.rand(10, 5))

    data = helpers.create_DataFrame()
    ds = DatasetWithMaskAndNoTarget(df=data)
    assert all(ds[-1][1] == torch.tensor([95, 96, 97, 98, 99], dtype=torch.int32))
    assert all(ds[-1][0] == torch.tensor([False, False, False, False, False]))