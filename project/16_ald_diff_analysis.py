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
import matplotlib.pyplot as plt

import vaep
import vaep.analyzers
import vaep.io.datasplits
# from vaep.analyzers import analyzers

import vaep.nb as config

logger = vaep.logging.setup_nb_logger()

# %%
folder_experiment = "runs/appl_ald_data/proteinGroups"
folder_data:str = '' # specify data directory if needed
fn_rawfile_metadata = "data/single_datasets/raw_meta.csv"

file_format = "pkl"
model_key='vae'

# %%
args = config.Config()
args.fn_rawfile_metadata = Path(fn_rawfile_metadata)
del fn_rawfile_metadata
args.folder_experiment = Path(folder_experiment)
del folder_experiment
args.file_format = file_format
del file_format
args = vaep.nb.add_default_paths(args, folder_data=folder_data)
del folder_data
args.model_key = model_key
# del model_key

# %%
data = vaep.io.datasplits.DataSplits.from_folder(args.data, file_format=args.file_format) 

# %%
observed = pd.concat([data.train_X, data.val_y, data.test_y])
observed

# %%
DATA_COMPLETENESS = 0.6
MIN_N_PROTEIN_GROUPS: int = 200
FRAC_PROTEIN_GROUPS: int = 0.622

from collections import namedtuple

def select_raw_data(df: pd.DataFrame, data_completeness: float, frac_protein_groups: int):
    msg = 'N samples: {}, M feat: {}'
    N, M = df.shape
    logger.info("Initally: " + msg.format(N, M))
    treshold_completeness = int(M * data_completeness)
    df = df.dropna(axis=1, thresh=treshold_completeness)
    logger.info(
        f"Dropped features quantified in less than {int(treshold_completeness)} samples.")
    N, M = df.shape
    logger.info("After feat selection: " + msg.format(N, M))
    min_n_protein_groups = int(M * frac_protein_groups)
    logger.info(
        f"Min No. of Protein-Groups in single sample: {min_n_protein_groups}")
    df = df.dropna(axis=0, thresh=MIN_N_PROTEIN_GROUPS)
    logger.info("Finally: " + msg.format(*df.shape))
    Cutoffs = namedtuple('Cutoffs', 'feat_completness_over_samples min_feat_in_sample')
    return df, Cutoffs(treshold_completeness, min_n_protein_groups)

ald_study, cutoffs = select_raw_data(observed.unstack(), data_completeness=DATA_COMPLETENESS, frac_protein_groups=FRAC_PROTEIN_GROUPS)


# %%
def plot_cutoffs(df, feat_completness_over_samples=None, min_feat_in_sample=None):
    notna = df.notna()
    fig, axes = plt.subplots(1, 2)
    ax  = axes[0]
    notna.sum(axis=0).sort_values().plot(rot=90, ax=ax, ylabel='count')
    if min_feat_in_sample is not None:
        ax.axhline(min_feat_in_sample)
    ax  = axes[1]
    notna.sum(axis=1).sort_values().plot(rot=90, ax=ax)
    if feat_completness_over_samples is not None:
        ax.axhline(feat_completness_over_samples)

plot_cutoffs(observed.unstack(), feat_completness_over_samples=cutoffs.feat_completness_over_samples, min_feat_in_sample=cutoffs.min_feat_in_sample)

# %% [markdown]
# ## load predictions for (real) missing data

# %%
list(args.out_preds.iterdir())

# %%
template = 'pred_real_na_{}.csv'

def load_pred(model_key):
    pred_real_na = pd.read_csv(args.out_preds / template.format(model_key))
    pred_real_na = pred_real_na.set_index(pred_real_na.columns[:-1].tolist())
    pred_real_na = pred_real_na.squeeze()
    pred_real_na.name = 'intensity'
    return pred_real_na



# %%
pred_real_na = load_pred(model_key=args.model_key)
pred_real_na.sample(3)


# %%
ax = pred_real_na.hist()

# %%
ax = observed.hist()

# %%
df = pd.concat([data.train_X, data.val_y, data.test_y, pred_real_na]).unstack()
df

# %%
assert df.isna().sum().sum() == 0, "DataFrame has missing entries"

# %%
