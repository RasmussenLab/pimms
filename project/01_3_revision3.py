# %% [markdown]
# # Compare models
#
# 1. Load available configurations
# 2. Load validation predictions
#     - calculate absolute error on common subset of data
#     - select top N for plotting by MAE from smallest (best) to largest (worst) (top N as specified, default 5)
#     - correlation per sample, correlation per feat, correlation overall
#     - MAE plots
# 3. Load test data predictions
#     - as for validation data
#     - top N based on validation data
#
# Model with `UNIQUE` key refer to samples uniquly split into training, validation and test data.
# These models could not use all sample for training. The predictions on simulated values
# are therefore restricted to the validation and test data from the set of unique samples.
# The models trained on all sample have additionally missing values in their training data,
# which were not missing in the unique samples. The comparison is therefore between models
# which had different data available for training.

# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import vaep
import vaep.imputation
import vaep.models
import vaep.nb
from vaep.analyzers import compare_predictions
from vaep.models.collect_dumps import select_content

pd.options.display.max_rows = 30
pd.options.display.min_rows = 10
pd.options.display.max_colwidth = 100

plt.rcParams.update({'figure.figsize': (3, 2)})
vaep.plotting.make_large_descriptors(7)

logger = vaep.logging.setup_nb_logger()
logging.getLogger('fontTools').setLevel(logging.WARNING)


def load_config_file(fname: Path, first_split='config_') -> dict:
    with open(fname) as f:
        loaded = yaml.safe_load(f)
    key = f"{select_content(fname.stem, first_split=first_split)}"
    return key, loaded


def build_text(s):
    ret = ''
    if not np.isnan(s["latent_dim"]):
        ret += f'LD: {int(s["latent_dim"])} '
    try:
        if len(s["hidden_layers"]):
            t = ",".join(str(x) for x in s["hidden_layers"])
            ret += f"HL: {t}"
    except TypeError:
        # nan
        pass
    return ret


# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% [markdown]
# Papermill script parameters:

# %% tags=["parameters"]
# files and folders
# Datasplit folder with data for experiment
folder_experiment: str = 'runs/example'
folder_data: str = ''  # specify data directory if needed
file_format: str = 'csv'  # change default to pickled files
# Machine parsed metadata from rawfile workflow
fn_rawfile_metadata: str = 'data/dev_datasets/HeLa_6070/files_selected_metadata_N50.csv'
models: str = 'Median,CF,DAE,VAE,KNN'  # picked models to compare (comma separated)
sel_models: str = ''  # user defined comparison (comma separated)
# Restrict plotting to top N methods for imputation based on error of validation data, maximum 10
plot_to_n: int = 5
feat_name_display: str = None  # display name for feature name (e.g. 'protein group')


# %%
models = 'KNN,KNN_UNIQUE'
folder_experiment = 'runs/rev3'

# %% [markdown]
# Some argument transformations

# %%
args = vaep.nb.get_params(args, globals=globals())
args

# %%
args = vaep.nb.args_from_dict(args)
args

# %%
figures = {}
dumps = {}

# %%
TARGET_COL = 'observed'
METRIC = 'MAE'
MIN_FREQ = None
MODELS_PASSED = args.models.split(',')
MODELS = MODELS_PASSED.copy()
FEAT_NAME_DISPLAY = args.feat_name_display
SEL_MODELS = None
if args.sel_models:
    SEL_MODELS = args.sel_models.split(',')

# %%


# %% [markdown]
# # Load predictions on validation and test data split
#

# %% [markdown]
# ## Validation data
# - set top N models to plot based on validation data split

# %%
pred_val = compare_predictions.load_split_prediction_by_modelkey(
    experiment_folder=args.folder_experiment,
    split='val',
    model_keys=MODELS_PASSED,
    shared_columns=[TARGET_COL])
SAMPLE_ID, FEAT_NAME = pred_val.index.names
if not FEAT_NAME_DISPLAY:
    FEAT_NAME_DISPLAY = FEAT_NAME
pred_val[MODELS]

# %%
pred_test = compare_predictions.load_split_prediction_by_modelkey(
    experiment_folder=args.folder_experiment,
    split='test',
    model_keys=MODELS_PASSED,
    shared_columns=[TARGET_COL])
pred_test

# %%
pred_val = pred_val.dropna()
pred_test = pred_test.dropna()

# %%
metrics = vaep.models.Metrics()
test_metrics = metrics.add_metrics(
    pred_test, key='test data')
test_metrics = pd.DataFrame(test_metrics)
test_metrics

# %%
metrics = vaep.models.Metrics()
val_metrics = metrics.add_metrics(
    pred_val, key='validation data')
val_metrics = pd.DataFrame(val_metrics)
val_metrics

# %%
