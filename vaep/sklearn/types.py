from dataclasses import dataclass
from collections import namedtuple

import pandas as pd
import sklearn

AucRocCurve = namedtuple("AucRocCurve", 'fpr tpr cutoffs')
PrecisionRecallCurve = namedtuple("PrecisionRecallCurve", 'precision recall cutoffs')


@dataclass
class ResultsSplit:
    auc: float = None
    roc: AucRocCurve = None
    prc: PrecisionRecallCurve = None


@dataclass
class Results:
    model: sklearn.base.BaseEstimator = None
    selected_features: list = None
    train: ResultsSplit = None
    test: ResultsSplit = None
    name: str = None


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
