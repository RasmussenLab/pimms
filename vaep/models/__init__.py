from functools import reduce
import logging
from operator import mul
from pathlib import Path
import pickle
import pprint
from typing import Tuple, List, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch 
from fastcore.foundation import L
from fastai import learner
import sklearn.metrics as sklm

from . import ae
from . import analysis
from . import collab

import vaep

logger = logging.getLogger(__name__)


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


def plot_training_losses(learner: learner.Learner,
                         name: str,
                         ax=None,
                         save_recorder: bool = True,
                         folder='figures',
                         figsize=(15, 8)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.set_title(f'{name} loss: Reconstruction loss')
    learner.recorder.plot_loss(skip_start=5, ax=ax)
    name = name.lower()
    _ = RecorderDump(learner.recorder, name).save(folder)
    vaep.savefig(fig, name=f'{name}_training',
                 folder=folder)
    return fig


def calc_net_weight_count(model: torch.nn.modules.module.Module) -> int:
    model.train()
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    weight_count = 0
    for param in model_params:
        weight_count += np.prod(param.size())
    return int(weight_count)

class RecorderDump:
    """Simple Class to hold fastai Recorder Callback data for serialization using pickle.
    """
    filename_tmp = 'recorder_{}.pkl'

    def __init__(self, recorder, name):
        self.losses = recorder.losses
        self.values = recorder.values
        self.iters = recorder.iters
        self.name = name

    def save(self, folder=Path('.')):
        with open(Path(folder) / self.filename_tmp.format(self.name), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath, name):
        with open(Path(filepath) / cls.filename_tmp.format(name), 'rb') as f:
            ret = pickle.load(f)
        return ret

    plot_loss = plot_loss




def split_prediction_by_mask(pred: pd.DataFrame,
                             mask: pd.DataFrame,
                             check_keeps_all: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        assert len(test_pred_real_na) + \
            len(test_pred_observed) == reduce(mul, pred.shape)
    return test_pred_observed, test_pred_real_na


def compare_indices(first_index: pd.Index, second_index: pd.Index) -> pd.Index:
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


scoring = [('MSE', sklm.mean_squared_error),
           ('MAE', sklm.mean_absolute_error)]


def get_metrics_df(pred_df: pd.DataFrame,
                   true_col: List[str] = None,
                   scoring: List[Tuple[str, Callable]] = scoring) -> pd.DataFrame:
    """Create metrics based on predictions, a truth reference and a
    list of scoring function with a name.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Prediction DataFrame containing `true_col`.
    true_col : List[str], optional
        Column of ground truth values, by default None
    scoring : List[Tuple[str, Callable]], optional
        List of tuples. A tuple is a set of (key, funtion) pairs.
        The function take y_true and y_pred - as for all sklearn metrics, by default scoring

    Returns
    -------
    pd.DataFrame
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    if not true_col:
        # assume first column is truth if None is given
        y_true = pred_df.iloc[:, 0]
        print(f'Selected as truth to compare to: {y_true.name}')
        y_pred = pred_df.iloc[:, 1:]
    else:
        if issubclass(type(true_col), int):
            y_true = pred_df.iloc[:, true_col]
            pred_df = pred_df.drop(y_true.name, axis=1)
        elif issubclass(type(true_col), str):
            y_true = pred_df[true_col]
            pred_df = pred_df.drop(true_col, axis=1)
        else:
            raise ValueError(
                f'true_col has to be of type str or int, not {type(true_col)}')
    if y_true.isna().any():
        raise ValueError(f"Ground truth column '{y_true.name}' contains missing values. "
                         "Drop these rows first.")
    # # If NAs in y_true should be allowed, the intersection between pred columns
    # # and y_true has to be added in the for loop below.
    # if y_true.isna().any():
        # logger.info(f"Remove {y_true.isna().sum()} from true_col {y_true.name}")
        # y_true = y_true.dropna()
        # assert len(y_true)
        
    metrics = {}
    for model_key in y_pred:
        model_pred = y_pred[model_key]
        model_pred_no_na = model_pred.dropna()
        if len(model_pred) > len(model_pred_no_na):
            logger.info(
                f"Drop indices for {model_key}: "
                "{}".format([(idx[0], idx[1])
                             for idx
                             in model_pred.index.difference(model_pred_no_na.index)]))

        metrics[model_key] = dict(
            [(k, float(f(y_true=y_true.loc[model_pred_no_na.index],
                         y_pred=model_pred_no_na)))
             for k, f in scoring]
        )
        metrics[model_key]['N'] = int(len(model_pred_no_na))
    # metrics = pd.DataFrame(metrics)
    return metrics


class Metrics():

    def __init__(self, no_na_key='no_na', with_na_key='with_na', na_column_to_drop=['interpolated']):
        self.no_na_key, self.with_na_key = no_na_key, with_na_key
        self.na_column_to_drop = na_column_to_drop
        self.metrics = {self.no_na_key: {}, self.with_na_key: {}}

    def add_metrics(self, pred, key):
        self.metrics[self.no_na_key][key] = get_metrics_df(
            pred_df=pred.dropna())
        mask_na = pred.isna().any(axis=1)
        # assert (~mask_na).sum() + mask_na.sum() == len(pred)
        if mask_na.sum():
            self.metrics[self.with_na_key][key] = get_metrics_df(
                pred_df=pred.loc[mask_na].drop(self.na_column_to_drop, axis=1))
        else:
            self.metrics[self.with_na_key][key] = None
        return {self.no_na_key: self.metrics[self.no_na_key][key],
                self.with_na_key:  self.metrics[self.with_na_key][key]}

    def __repr__(self):
        return pprint.pformat(self.metrics, indent=2, compact=True)
