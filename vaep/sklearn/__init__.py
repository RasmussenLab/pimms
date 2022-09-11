import sklearn

from mrmr import mrmr_classif

from .types import Splits, ResultsSplit, Results, AucRocCurve, PrecisionRecallCurve


def run_model(splits: Splits,
              model: sklearn.base.BaseEstimator = sklearn.linear_model.LogisticRegression,
              ) -> Results:
    selected_features = mrmr_classif(X=splits.X_train, y=splits.y_train, K=10)

    model = model()
    model.fit(splits.X_train[selected_features], splits.y_train)

    results_train = ResultsSplit(auc=model.score(
        splits.X_train[selected_features], splits.y_train))
    results_test = ResultsSplit(auc=model.score(
        splits.X_test[selected_features], splits.y_test))

    results_test.auc = model.score(
        splits.X_test[selected_features], splits.y_test)

    pred_score_test = model.predict_proba(
        splits.X_test[selected_features])[:, 1]
    results_test.roc = AucRocCurve(
        *sklearn.metrics.roc_curve(y_true=splits.y_test, y_score=pred_score_test))
    results_test.prc = PrecisionRecallCurve(*sklearn.metrics.precision_recall_curve(
        y_true=splits.y_test, probas_pred=pred_score_test))

    results_test.aps = sklearn.metrics.average_precision_score(
        y_true=splits.y_test,
         y_score=pred_score_test)

    ret = Results(model=model,
                  selected_features=selected_features,
                  train=results_train,
                  test=results_test)
    return ret
