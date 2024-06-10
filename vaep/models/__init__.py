import json
import logging
import pickle
import pprint
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sklm
import torch
from fastai import learner
from fastcore.foundation import L

import vaep
from vaep.models import ae, analysis, collab, vae

logger = logging.getLogger(__name__)

NUMPY_ONE = np.int64(1)

__all__ = ['ae', 'analysis', 'collab', 'vae', 'plot_loss', 'plot_training_losses',
           'calc_net_weight_count', 'RecorderDump', 'split_prediction_by_mask',
           'compare_indices', 'collect_metrics', 'calculte_metrics',
           'Metrics', 'get_df_from_nested_dict']


def plot_loss(recorder: learner.Recorder,
              norm_train: np.int64 = NUMPY_ONE,
              norm_val: np.int64 = NUMPY_ONE,
              skip_start: int = 5,
              with_valid: bool = True,
              ax: plt.Axes = None) -> plt.Axes:
    """Adapted Recorder.plot_loss to accept matplotlib.axes.Axes argument.
    Allows to build combined graphics.

    Parameters
    ----------
    recorder : learner.Recorder
        fastai Recorder object, learn.recorder
    norm_train: np.int64, optional
        Normalize epoch loss by number of training samples, by default 1
    norm_val: np.int64, optional
        Normalize epoch loss by number of validation samples, by default 1
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
        _, ax = plt.subplots()
    ax.plot(list(range(skip_start, len(recorder.losses))),
            recorder.losses[skip_start:] / norm_train, label='train')
    if with_valid:
        idx = (np.array(recorder.iters) < skip_start).sum()
        ax.plot(recorder.iters[idx:], L(
            recorder.values[idx:]).itemgot(1) / norm_val, label='valid')
        ax.legend()
    return ax


NORM_ONES = np.array([1, 1], dtype='int')


def plot_training_losses(learner: learner.Learner,
                         name: str,
                         ax=None,
                         norm_factors=NORM_ONES,
                         folder='figures',
                         figsize=(15, 8)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.set_title(f'{name} loss')
    norm_train, norm_val = norm_factors  # exactly two
    with_valid = True
    if norm_val is None:
        with_valid = False
    learner.recorder.plot_loss(skip_start=5, ax=ax, with_valid=with_valid,
                               norm_train=norm_train, norm_val=norm_val)
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

    def save(self, folder='.'):
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


def collect_metrics(metrics_jsons: List, key_fct: Callable) -> dict:
    """Collect and aggregate a bunch of json metrics.

    Parameters
    ----------
    metrics_jsons : List
        list of filepaths to json metric files
    key_fct : Callable
        Callable which creates key function of a single filepath

    Returns
    -------
    dict
        Aggregated metrics dictionary with outer key defined by key_fct

    Raises
    ------
    AssertionError:
        If key should be overwritten, but value would change.
    """
    all_metrics = {}
    for fname in metrics_jsons:
        fname = Path(fname)
        logger.info(f"Load file: {fname = }")

        key = key_fct(fname)  # level, repeat

        logger.debug(f"{key = }")
        with open(fname) as f:
            loaded = json.load(f)
        loaded = vaep.pandas.flatten_dict_of_dicts(loaded)

        if key not in all_metrics:
            all_metrics[key] = loaded
            continue
        for k, v in loaded.items():
            if k in all_metrics[key]:
                logger.debug(f"Found existing key: {k = } ")
                assert all_metrics[key][k] == v, "Diverging values for {k}: {v1} vs {v2}".format(
                    k=k,
                    v1=all_metrics[key][k],
                    v2=v)
            else:
                all_metrics[key][k] = v
    return all_metrics


def calculte_metrics(pred_df: pd.DataFrame,
                     true_col: List[str] = None,
                     scoring: List[Tuple[str, Callable]] = scoring) -> dict:
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
            y_pred = pred_df.drop(y_true.name, axis=1)
        elif issubclass(type(true_col), str):
            y_true = pred_df[true_col]
            y_pred = pred_df.drop(true_col, axis=1)
        else:
            raise ValueError(
                f'true_col has to be of type str or int, not {type(true_col)}')
    if y_true.isna().any():
        raise ValueError(f"Ground truth column '{y_true.name}' contains missing values. "
                         "Drop these rows first.")

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
        metrics[model_key]['prop'] = len(model_pred_no_na) / len(model_pred)
    return metrics


class Metrics():

    def __init__(self):
        self.metrics = {}

    def add_metrics(self, pred, key):
        self.metrics[key] = calculte_metrics(pred_df=pred.dropna())
        return self.metrics[key]

    def __repr__(self):
        return pprint.pformat(self.metrics, indent=2, compact=True)


def get_df_from_nested_dict(nested_dict,
                            column_levels=(
                                'data_split', 'model', 'metric_name'),
                            row_name='subset'):
    metrics = {}
    for k, run_metrics in nested_dict.items():
        metrics[k] = vaep.pandas.flatten_dict_of_dicts(run_metrics)

    metrics = pd.DataFrame.from_dict(metrics, orient='index')
    metrics.columns.names = column_levels
    metrics.index.name = row_name
    return metrics
