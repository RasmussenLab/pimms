# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
import pandas as pd

from vaep.models.collect_dumps import collect_metrics

# %%
all_metrics = collect_metrics(snakemake.input)
all_metrics

# %%
fname = Path(snakemake.output.out)
all_metrics = pd.DataFrame(all_metrics)
all_metrics.to_json(fname)
all_metrics

# %%
df_metrics_long = all_metrics.set_index('id')
df_metrics_long.columns = pd.MultiIndex.from_tuples(df_metrics_long.columns)
df_metrics_long.columns.names = ['data_split', 'model', 'metric_name']
df_metrics_long.stack('model')

# %%
fname = fname.with_suffix('.csv')
df_metrics_long.to_csv(fname)
# pd.read_csv(fname, index_col=0, header=[0, 1, 2, 3]).stack('model')
