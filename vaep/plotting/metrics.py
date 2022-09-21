import matplotlib
import pandas as pd

from vaep.sklearn.types import ResultsSplit


def plot_split_auc(result: ResultsSplit, name: str, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    col_name = f"{name} (auc: {result.auc:.3f})"
    roc = pd.DataFrame(result.roc, index='fpr tpr cutoffs'.split()
                       ).rename({'tpr': col_name})
    ax = roc.T.plot('fpr', col_name,
                    xlabel='false positive rate',
                    ylabel='true positive rate',
                    ax=ax)
    return ax


def plot_split_prc(result: ResultsSplit, name: str, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    col_name = f"{name} (aps: {result.aps:.3f})"
    roc = pd.DataFrame(result.prc, index='precision recall cutoffs'.split()
                       ).rename({'precision': col_name})
    ax = roc.T.plot('recall', col_name,
                    xlabel='true positive rate',
                    ylabel='precision',
                    ax=ax)
    return ax
