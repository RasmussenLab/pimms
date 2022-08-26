import pandas as pd


def calc_errors(pred: pd.DataFrame, freq_feat: pd.Series, target_col='observed') -> pd.DataFrame:
    """Calculate absolute errors and sort by freq of features."""
    errors = pred.drop(target_col, axis=1).sub(pred[target_col], axis=0)
    errors = errors.abs().groupby(freq_feat.index.name).mean()  # absolute error
    errors = errors.join(freq_feat)
    errors = errors.sort_values(by=freq_feat.name, ascending=True)
    errors.columns.name = 'model'
    return errors
