# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
import pandas as pd
pd.options.display.max_columns = 30 

from vaep.models.collect_dumps import collect_configs
from vaep.logging import setup_nb_logger
logger = setup_nb_logger()

# %%
snakemake.input[:10]

# %%
all_configs = collect_configs(snakemake.input)
all_config = pd.DataFrame(all_configs).set_index('id')
all_config

# %%
fname = Path(snakemake.output.out)
all_config.reset_index().to_json()
all_config = all_config.set_index('model', append=True)
all_config.to_csv(fname.with_suffix('.csv'))
all_config
