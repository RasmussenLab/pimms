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
import json
from pathlib import Path
import pandas as pd
import vaep.pandas


# %%
def select_content(s:str):
    s = s.split('metrics_')[1]
    assert isinstance(s, str), f"More than one split: {s}"
    entries = s.split('_')
    if len(entries) > 1:
        s = '_'.join(entries[:-1])
    return s
    
test_cases = ['model_metrics_HL_1024_512_256_dae',
              'model_metrics_HL_1024_512_vae',
              'model_metrics_collab']
 
for test_case in test_cases:
    print(f"{test_case} = {select_content(test_case)}")

# %%
all_metrics = {}
for fname in snakemake.input:
    fname = Path(fname)
    # "grandparent" directory gives name beside name of file
    key = f"{fname.parents[1].name}_{select_content(fname.stem)}"
    print(f"{key = }")
    with open(fname) as f:
        loaded = json.load(f)
    loaded = vaep.pandas.flatten_dict_of_dicts(loaded)
    if key not in all_metrics:
        all_metrics[key] = loaded
        continue
    for k, v in loaded.items():
        if k in all_metrics[key]:
            assert all_metrics[key][k] == v, "Diverging values for {k}: {v1} vs {v2}".format(
                k=k,
                v1=all_metrics[key][k],
                v2=v)
        else:
            all_metrics[key][k] = v

pd.DataFrame(all_metrics).to_json(snakemake.output.out)

# %%
all_metrics

# %%
