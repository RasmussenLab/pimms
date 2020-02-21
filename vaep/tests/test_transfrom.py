import pandas as pd
import numpy as np 
from vaep.transform import log

def test_log():
    row = pd.Series([np.NaN, 0.0, np.exp(1), np.exp(2)])
    row = log(row)
    assert row.equals(pd.Series([np.NaN, np.NaN, 1.0, 2.0]))
