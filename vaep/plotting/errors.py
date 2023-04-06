"""Plot errors based on DataFrame with model predictions."""
import pandas as pd
from matplotlib.axes import Axes
import seaborn as sns

import vaep.pandas.calc_errors


def plot_errors_binned(pred: pd.DataFrame, target_col='observed', ax: Axes = None) -> Axes:
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
                     x='bin', y='intensity', hue='model')
    ax.xaxis.set_tick_params(rotation=-90)
    return ax, errors_binned
