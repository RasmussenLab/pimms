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
import pandas as pd
pd.options.display.max_columns = 30 

from vaep.models.collect_dumps import collect_configs
from vaep.logging import setup_nb_logger
logger = setup_nb_logger()

# %%

all_configs = collect_configs(snakemake.input)
all_config = pd.DataFrame(all_configs)
all_config.T

# %%
all_config.to_json(snakemake.output.out)

# %%
