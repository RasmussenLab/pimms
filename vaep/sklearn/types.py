"""Types used in scikit-learn pipelines."""
import pickle
from dataclasses import dataclass
from collections import namedtuple

import pandas as pd
import sklearn

AucRocCurve = namedtuple("AucRocCurve", 'fpr tpr cutoffs')
PrecisionRecallCurve = namedtuple("PrecisionRecallCurve", 'precision recall cutoffs')


@dataclass
class ResultsSplit:
    auc: float = None  # receiver operation curve area under the curve
    aps: float = None  # average precision score
    roc: AucRocCurve = None
    prc: PrecisionRecallCurve = None


@dataclass
class Results:
    model: sklearn.base.BaseEstimator = None
    selected_features: list = None
    train: ResultsSplit = None
    test: ResultsSplit = None
    name: str = None

    def to_pickle(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    # not necessary, but here it ensure Results is defined
    @classmethod
    def from_pickle(cls, fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
