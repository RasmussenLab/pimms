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
# start coding here
import yaml
from pathlib import Path
import vaep.pandas
import pandas as pd
pd.options.display.max_columns = 30 
all_configs = {}


# %%
def select_content(s:str, first_split='metrics_'):
    s = s.split(first_split)[1]
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
for fname in snakemake.input:
    fname = Path(fname)
    # "grandparent" directory gives name beside name of file
    key = f"{fname.parents[1].name}_{select_content(fname.stem, 'config_')}"
    print(f"{key = }")
    with open(fname) as f:
        loaded = yaml.safe_load(f)   
    if key not in all_configs:
        all_configs[key] = loaded
        continue
    for k, v in loaded.items():
        if k in all_configs[key]:
            if not all_configs[key][k] == v:
                print(
                    "Diverging values for {k}: {v1} vs {v2}".format(
                k=k,
                v1=all_configs[key][k],
                v2=v)
                )
        else:
            all_configs[key][k] = v

# %%
all_config = pd.DataFrame(all_configs)
all_config.T

# %%
all_config.to_json(snakemake.output.out)
