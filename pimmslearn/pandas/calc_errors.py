import pandas as pd


def calc_errors_per_feat(pred: pd.DataFrame, freq_feat: pd.Series, target_col='observed') -> pd.DataFrame:
    """Calculate absolute errors and sort by freq of features."""
    n_obs = pred.groupby(freq_feat.index.name)[
        target_col].count().rename('n_obs')
    errors = pred.drop(target_col, axis=1).sub(pred[target_col], axis=0)
    errors = errors.abs().groupby(freq_feat.index.name).mean()  # absolute error
    errors = errors.join(freq_feat).join(n_obs)
    errors = errors.sort_values(by=freq_feat.name, ascending=True)
    errors.columns.name = 'model'
    return errors


def calc_errors_per_bin(pred: pd.DataFrame, target_col='observed') -> pd.DataFrame:
    """Calculate absolute errors. Bin by integer value of simulated NA and provide count."""
    errors = get_absolute_error(pred, y_true=target_col)
    errors['bin'] = pred[target_col].astype(int)  # integer bin of simulated NA
    n_obs = errors.groupby('bin').size().rename('n_obs')
    errors = errors.join(n_obs, on='bin')
    errors = errors.sort_values(by='n_obs', ascending=True)
    errors.columns.name = 'model'
    return errors


def get_absolute_error(pred: pd.DataFrame, y_true: str = 'observed') -> pd.DataFrame:
    errors = pred.drop(y_true, axis=1).sub(pred[y_true], axis=0)
    return errors.abs()
