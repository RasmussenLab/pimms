from functools import reduce
from operator import mul
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastcore.foundation import L
from fastai import learner

from . import ae


def plot_loss(recorder: learner.Recorder,
              skip_start: int = 5,
              with_valid: bool = True,
              ax: plt.Axes = None) -> plt.Axes:
    """Adapted Recorder.plot_loss to accept matplotlib.axes.Axes argument.
    Allows to build combined graphics.

    Parameters
    ----------
    recorder : learner.Recorder
        fastai Recorder object, learn.recorder
    skip_start : int, optional
        Skip N first batch metrics, by default 5
    with_valid : bool, optional
        Add validation data loss, by default True
    ax : plt.Axes, optional
        Axes to plot on, by default None

    Returns
    -------
    plt.Axes
        [description]
    """
    if not ax:
        fig, ax = plt.subplots()
    ax.plot(list(range(skip_start, len(recorder.losses))),
            recorder.losses[skip_start:], label='train')
    if with_valid:
        idx = (np.array(recorder.iters) < skip_start).sum()
        ax.plot(recorder.iters[idx:], L(
            recorder.values[idx:]).itemgot(1), label='valid')
        ax.legend()
    return ax


def split_prediction_by_mask(pred:pd.DataFrame, 
                             mask:pd.DataFrame, 
                             check_keeps_all:bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """[summary]

    Parameters
    ----------
    pred : pd.DataFrame
        prediction DataFrame
    mask : pd.DataFrame
        Mask with same indices as pred DataFrame.
    check_keeps_all : bool, optional
        if True, perform sanity checks, by default False

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        prediction for inversed mask, and predicitions for mask
    """
    test_pred_observed = pred[~mask].stack()
    test_pred_real_na = pred[mask].stack()
    if check_keeps_all:
        assert len(test_pred_real_na) + len(test_pred_observed) == reduce(mul, pred.shape)
    return test_pred_observed, test_pred_real_na


def compare_indices(first_index:pd.Index, second_index:pd.Index) -> pd.Index:
    """Show difference of indices in other index wrt. to first. First should be the larger
    collection wrt to the second. This is the set difference of two Index objects.
    
    If second index is a superset of indices of the first, the set will be empty,
    although there  are differences (default behaviour in pandas).
    
    Parameters
    ----------
    first_index : pd.Index
        Index, should be superset
    second_index : pd.Index
        Index, should be the subset
        
    Returns
    -------
    pd.Index
        Return a new Index with elements of the first index not in second.
    """
    _diff_index = first_index.difference(second_index)
    if len(_diff_index):   
        print("Some predictions couldn't be generated using the approach using artifical replicates.\n"
             "These will be omitted for evaluation.")
        for _index in _diff_index:
            print(f"{_index[0]:<40}\t {_index[1]:<40}")
    return _diff_index