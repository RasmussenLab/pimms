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

# %% [markdown]
# # Collect config of model
#
# - dumped arguments of all model runs

# %%
from pathlib import Path
import pandas as pd

from vaep.logging import setup_nb_logger
from vaep.models.collect_dumps import collect_configs

pd.options.display.max_columns = 30

logger = setup_nb_logger()

# %%
snakemake.input[:10]

# %%
all_configs = collect_configs(snakemake.input)
df_config = pd.DataFrame(all_configs).set_index('id')
df_config

# %%
fname = Path(snakemake.output.out)
fname

# %%
df_config.reset_index().to_json()
df_config = df_config.set_index('model', append=True)
df_config.to_csv(fname.with_suffix('.csv'))
df_config
