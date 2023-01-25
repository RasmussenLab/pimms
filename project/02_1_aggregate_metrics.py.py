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

from vaep.models.collect_dumps import collect_metrics

# %%
all_metrics = collect_metrics(snakemake.input)

all_metrics = pd.DataFrame(all_metrics)
all_metrics.to_json(snakemake.output.out)
all_metrics

# %%
all_metrics
