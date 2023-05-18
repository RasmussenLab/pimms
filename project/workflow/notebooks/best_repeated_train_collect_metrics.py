# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import json
from pathlib import Path
import pandas as pd
import vaep.models.collect_dumps

REPITITION_NAME = snakemake.params.repitition_name


def select_content(s: str):
    s = s.split("metrics_")[1]
    assert isinstance(s, str), f"More than one split: {s}"
    model, repeat = s.split("_")
    return model, int(repeat)

def key_from_fname(fname: Path):
    _, repeat = select_content(fname.stem)
    key = (fname.parent.name, repeat)
    return key

def load_metric_file(fname: Path, frist_split='metrics'):
    fname = Path(fname)
    with open(fname) as f:
        loaded = json.load(f)
    loaded = vaep.pandas.flatten_dict_of_dicts(loaded)
    key = key_from_fname(fname) # '_'.join(key_from_fname(fname))
    return key, loaded

load_metric_file(snakemake.input.metrics[0])

# %%


all_metrics = vaep.models.collect_dumps.collect(snakemake.input.metrics, load_metric_file)
metrics = pd.DataFrame(all_metrics)
metrics = metrics.set_index('id')
metrics.index = pd.MultiIndex.from_tuples(
                            metrics.index,
                            names=("data level", REPITITION_NAME))
metrics.columns = pd.MultiIndex.from_tuples(
                                metrics.columns, 
                                names=('data_split', 'model', 'metric_name'))
metrics = (metrics
            .stack(['metric_name', 'model'])
            .unstack(['model', 'metric_name'])
            .T)
metrics

# %%
metrics = metrics.loc[
    pd.IndexSlice[
        ["valid_fake_na", "test_fake_na"],
        ["CF", "DAE", "VAE"],
        :]
]
metrics

# %%
FOLDER = Path(snakemake.params.folder)
fname = FOLDER / "metrics.pkl" 
metrics.to_csv(fname.with_suffix(".csv"))
metrics.to_excel(fname.with_suffix(".xlsx"))
metrics.to_pickle(fname)
fname
