
import matplotlib.pyplot as plt
import numpy as np

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
