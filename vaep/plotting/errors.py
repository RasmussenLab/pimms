"""Plot errors based on DataFrame with model predictions."""
import pandas as pd
from matplotlib.axes import Axes
import seaborn as sns

import vaep.pandas.calc_errors


def plot_errors_binned(pred: pd.DataFrame, target_col='observed',
                       ax: Axes = None,
                       palette:dict=None) -> Axes:
    assert target_col in pred.columns, f'Specify `target_col` parameter, `pred` do no contain: {target_col}'
    models_order = pred.columns.to_list()
    models_order.remove(target_col)
    errors_binned = vaep.pandas.calc_errors.calc_errors_per_bin(
        pred=pred, target_col=target_col)

    meta_cols = ['bin', 'n_obs']  # calculated along binned error
    n_obs = (errors_binned[meta_cols]
             .apply(
        lambda x: f"{x.bin} (N={x.n_obs:,d})", axis=1
    )
        .rename('bin')
        .astype('category')
    )

    errors_binned = (errors_binned[models_order]
                     .stack()
                     .to_frame('intensity')
                     .join(n_obs)
                     .reset_index()
                     )

    ax = sns.barplot(data=errors_binned, ax=ax,
                     x='bin', y='intensity', hue='model',
                     palette=palette)
    ax.xaxis.set_tick_params(rotation=-90)
    return ax, errors_binned


def plot_rolling_error(errors: pd.DataFrame, metric_name: str, window: int = 200,
                       min_freq=None, freq_col: str = 'freq', colors_to_use=None,
                       ax=None):
    errors_smoothed = errors.drop(freq_col, axis=1).rolling(window=window, min_periods=1).mean()
    errors_smoothed_max = errors_smoothed.max().max()
    errors_smoothed[freq_col] = errors[freq_col]
    if min_freq is None:
        min_freq=errors_smoothed[freq_col].min()
    else:
        errors_smoothed = errors_smoothed.loc[errors_smoothed[freq_col] > min_freq]
    ax = errors_smoothed.plot(x=freq_col, ylabel=f'rolling average error ({metric_name})',
                              color=colors_to_use,
                              xlim=(min_freq, errors_smoothed[freq_col].max()),
                              ylim=(0, min(errors_smoothed_max, 5)), 
                              ax=None)
    return ax
