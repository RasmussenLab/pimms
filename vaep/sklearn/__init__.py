"""Scikit-learn related functions for the project for ALD part.

Might be moved to a separate package in the future.
"""
import pandas as pd
import sklearn
import sklearn.model_selection

from mrmr import mrmr_classif

from .types import Splits, ResultsSplit, Results, AucRocCurve, PrecisionRecallCurve

default_model: sklearn.linear_model.LogisticRegression = sklearn.linear_model.LogisticRegression(
    random_state=42,
    solver='liblinear')


def run_model(splits: Splits,
              model: sklearn.base.BaseEstimator = default_model,
              n_feat_to_select=9,
              ) -> Results:
    selected_features = mrmr_classif(X=splits.X_train, y=splits.y_train, K=n_feat_to_select)

    model.fit(splits.X_train[selected_features], splits.y_train)

    pred_score_test = model.predict_proba(
        splits.X_test[selected_features])[:, 1]
    results_test = get_results_split(y_true=splits.y_test, y_score=pred_score_test)

    pred_score_train = model.predict_proba(
        splits.X_train[selected_features])[:, 1]
    results_train = get_results_split(y_true=splits.y_train, y_score=pred_score_train)

    ret = Results(model=model,
                  selected_features=selected_features,
                  train=results_train,
                  test=results_test)
    return ret


def get_results_split(y_true, y_score):
    ret = ResultsSplit(auc=sklearn.metrics.roc_auc_score(
        y_true=y_true, y_score=y_score))

    ret.roc = AucRocCurve(
        *sklearn.metrics.roc_curve(y_true=y_true, y_score=y_score))
    ret.prc = PrecisionRecallCurve(*sklearn.metrics.precision_recall_curve(
        y_true=y_true, probas_pred=y_score))

    ret.aps = sklearn.metrics.average_precision_score(
        y_true=y_true,
        y_score=y_score)
    return ret


scoring_defaults = ['precision', 'recall', 'f1', 'balanced_accuracy', 'roc_auc']


def find_n_best_features(X, y, name,
                         model=default_model,
                         groups=None,
                         n_features_max=15,
                         random_state=42,
                         scoring=scoring_defaults):
    summary = []
    cv = sklearn.model_selection.RepeatedStratifiedKFold(
        n_splits=5, n_repeats=10, random_state=random_state)
    in_both = y.index.intersection(X.index)
    # could have a warning in case
    _X = X.loc[in_both]
    _y = y.loc[in_both]
    for n_features in range(1, n_features_max + 1):
        selected_features = mrmr_classif(_X, _y, K=n_features)
        _X_mrmr = _X[selected_features]
        scores = sklearn.model_selection.cross_validate(
            estimator=model, X=_X_mrmr, y=_y, groups=groups, scoring=scoring, cv=cv)
        scores['n_features'] = n_features
        scores['test_case'] = name
        scores['n_observations'] = _X.shape[0]
        results = pd.DataFrame(scores)
        summary.append(results)
    summary_n_features = pd.concat(summary)
    return summary_n_features
