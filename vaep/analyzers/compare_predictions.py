import pandas as pd
from typing import List


def load_predictions(pred_files: List, shared_columns=['observed', 'interpolated']):

    pred_files = iter(pred_files)
    fname = next(pred_files)
    pred = pd.read_csv(fname, index_col=[0, 1])

    for fname in pred_files:
        _pred_file = pd.read_csv(fname, index_col=[0, 1])
        if shared_columns:
            assert all(pred[shared_columns] == _pred_file[shared_columns])
            pred = pred.join(_pred_file.drop(shared_columns, axis=1))
        else:
            pred = pred.join(_pred_file)
    return pred
