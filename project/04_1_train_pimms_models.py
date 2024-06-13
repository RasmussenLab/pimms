# %% [markdown]
# # PIMMS Tutorial: Scikit-learn style transformers
#
# 1. Load data into pandas dataframe
# 2. Fit model on training data, potentially specify validation data
# 3. Impute only missing values with predictions from model
#
# Autoencoders need wide training data, i.e. a sample with all its features' intensities, whereas
# Collaborative Filtering needs long training data, i.e. sample identifier a feature identifier and the intensity.
# Both data formats can be transformed into each other, but models using long data format do not need to
# take care of missing values.

# %%
import os
from importlib import metadata

IN_COLAB = 'COLAB_GPU' in os.environ
if IN_COLAB:
    try:
        _v = metadata.version('pimms-learn')
        print(f"Running in colab and pimms-learn ({_v}) is installed.")
    except metadata.PackageNotFoundError:
        print("Install PIMMS...")
        # # !pip install git+https://github.com/RasmussenLab/pimms.git@dev
        # !pip install pimms-learn

# %% [markdown]
# If on colab, please restart the environment and run everything from here on.
#
# Specify example data:

# %%
import os

IN_COLAB = 'COLAB_GPU' in os.environ

fn_intensities = 'data/dev_datasets/HeLa_6070/protein_groups_wide_N50.csv'
if IN_COLAB:
    fn_intensities = ('https://raw.githubusercontent.com/RasmussenLab/pimms/main/'
                      'project/data/dev_datasets/HeLa_6070/protein_groups_wide_N50.csv')

# %% [markdown]
# Load package.

# %%
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

import vaep.filter
import vaep.plotting.data
import vaep.sampling
from vaep.plotting.defaults import color_model_mapping
from vaep.sklearn.ae_transformer import AETransformer
from vaep.sklearn.cf_transformer import CollaborativeFilteringTransformer

vaep.plotting.make_large_descriptors(8)


logger = logger = vaep.logging.setup_nb_logger()
logging.getLogger('fontTools').setLevel(logging.ERROR)

# %% [markdown]
# ## Parameters
# Can be set by papermill on the command line or manually in the (colab) notebook.

# %%
fn_intensities: str = fn_intensities  # path or url to the data file in csv format
index_name: str = 'Sample ID'  # name of the index column
column_name: str = 'protein group'  # name of the column index
select_features: bool = True  # Whether to select features based on prevalence
feat_prevalence: float = 0.2  # minimum prevalence of a feature to be included
sample_completeness: float = 0.3  # minimum completeness of a sample to be included
sample_splits: bool = True  # Whether to sample validation and test data
frac_non_train: float = 0.1  # fraction of non training data (validation and test split)
frac_mnar: float = 0.0  # fraction of missing not at random data, rest: missing completely at random
random_state: int = 42  # random state for reproducibility

# %% [markdown]
# ## Data

# %%
df = pd.read_csv(fn_intensities, index_col=0)
df.head()

# %% [markdown]
# We will need the data in long format for Collaborative Filtering.
# Naming both the row and column index assures
# that the data can be transformed very easily into long format:

# %%
df.index.name = index_name  # already set
df.columns.name = column_name  # not set due to csv disk file format
df.head()

# %% [markdown]
# ### Data transformation: log2 transformation
# Transform the data using the logarithm, here using base 2:

# %%
df = np.log2(df + 1)
df.head()


# %% [markdown]
# ### two plots inspecting data availability
#
# 1. proportion of missing values per feature median (N = protein groups)
# 2. CDF of available intensities per protein group

# %%
ax = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=df, type='boxplot')


# %%
df.notna().sum().sort_values().plot()


# %% [markdown]
# ### Data selection
# define a minimum feature and sample frequency for a feature to be included

# %%
if select_features:
    # potentially this can take a few iterations to stabilize.
    df = vaep.filter.select_features(df, feat_prevalence=feat_prevalence)
    df = vaep.filter.select_features(df=df, feat_prevalence=sample_completeness, axis=1)
df.shape


# %% [markdown]
# Transform to long-data format:

# %%
df = df.stack().to_frame('intensity')
df

# %% [markdown]
# ## Optionally: Sample data
# - models can be trained without subsetting the data
# - allows evaluation of the models

# %%
if sample_splits:
    splits, thresholds, fake_na_mcar, fake_na_mnar = vaep.sampling.sample_mnar_mcar(
        df_long=df,
        frac_non_train=frac_non_train,
        frac_mnar=frac_mnar,
        random_state=random_state,
    )
    splits = vaep.sampling.check_split_integrity(splits)
else:
    splits = vaep.sampling.DataSplits(is_wide_format=False)
    splits.train_X = df

# %% [markdown]
# The resulting DataFrame with one column has an `MulitIndex` with the sample and feature identifier.

# %% [markdown]
# ## Collaborative Filtering
#
# Inspect annotations of the scikit-learn like Transformer:

# %%
# # CollaborativeFilteringTransformer?

# %% [markdown]
# Let's set up collaborative filtering without a validation or test set, using
# all the data there is.

# %%
cf_model = CollaborativeFilteringTransformer(
    target_column='intensity',
    sample_column='Sample ID',
    item_column='protein group',
    out_folder='runs/scikit_interface')

# %% [markdown]
# We use `fit` and `transform` to train the model and impute the missing values.
# > Scikit learns interface requires a `X` and `y`. `y` is the validation data in our context.
# > We might have to change the interface to allow usage within pipelines (-> `y` is not needed).
# > This will probably mean setting up a validation set within the model.

# %%
cf_model.fit(splits.train_X,
             splits.val_y,
             cuda=False,
             epochs_max=20,
             )

# %% [markdown]
# Impute missing values usin `transform` method:

# %%
df_imputed = cf_model.transform(df).unstack()
assert df_imputed.isna().sum().sum() == 0
df_imputed.head()

# %% [markdown]
# Let's plot the distribution of the imputed values vs the ones used for training:

# %%
df_imputed = df_imputed.stack()  # long-format
observed = df_imputed.loc[df.index]
imputed = df_imputed.loc[df_imputed.index.difference(df.index)]
df_imputed = df_imputed.unstack()  # back to wide-format
# some checks
assert len(df) == len(observed)
assert df_imputed.shape[0] * df_imputed.shape[1] == len(imputed) + len(observed)

fig, axes = plt.subplots(2, figsize=(8, 4))

min_max = vaep.plotting.data.get_min_max_iterable(
    [observed, imputed])
label_template = '{method} (N={n:,d})'
ax, _ = vaep.plotting.data.plot_histogram_intensities(
    observed,
    ax=axes[0],
    min_max=min_max,
    label=label_template.format(method='measured',
                                n=len(observed),
                                ),
    color='grey',
    alpha=1)
_ = ax.legend()
ax, _ = vaep.plotting.data.plot_histogram_intensities(
    imputed,
    ax=axes[1],
    min_max=min_max,
    label=label_template.format(method='CF imputed',
                                n=len(imputed),
                                ),
    color=color_model_mapping['CF'],
    alpha=1)
_ = ax.legend()

# %% [markdown]
# ## AutoEncoder architectures

# %%
# Use wide format of data
splits.to_wide_format()
splits.train_X

# %% [markdown]
# Validation data for early stopping (if specified)

# %%
splits.val_y

# %% [markdown]
# Training and validation need the same shape:

# %%
if splits.val_y is not None:
    splits.val_y = pd.DataFrame(pd.NA, index=splits.train_X.index,
                                columns=splits.train_X.columns).fillna(splits.val_y)

    print(splits.train_X.shape, splits.val_y.shape)

# %% [markdown]
# Select either `DAE` or `VAE` model by chance:

# %%
model_selected = random.choice(['DAE', 'VAE'])
print("Selected model by chance:", model_selected)
model = AETransformer(
    model=model_selected,
    hidden_layers=[512,],
    latent_dim=50,
    out_folder='runs/scikit_interface',
    batch_size=10,
)

# %%
model.fit(splits.train_X, splits.val_y,
          epochs_max=50,
          cuda=False)

# %% [markdown]
# Impute missing values using `transform` method:

# %%
df_imputed = model.transform(splits.train_X).stack()
df_imputed

# %% [markdown]
# Evaluate the model using the validation data:

# %%
if splits.val_y is not None:
    pred_val = splits.val_y.stack().to_frame('observed')
    pred_val[model_selected] = df_imputed
    val_metrics = vaep.models.calculte_metrics(pred_val, 'observed')
    display(val_metrics)

    fig, ax = plt.subplots(figsize=(8, 2))

    ax, errors_binned = vaep.plotting.errors.plot_errors_by_median(
        pred=pred_val,
        target_col='observed',
        feat_medians=splits.train_X.median(),
        ax=ax,
        metric_name='MAE',
        palette=color_model_mapping)

# %% [markdown]
# replace predicted values with validation data values

# %%
splits.to_long_format()
df_imputed = df_imputed.replace(splits.val_y).replace(splits.test_y)
df_imputed

# %% [markdown]
# Plot the distribution of the imputed values vs the observed data:

# %%
observed = df_imputed.loc[df.index].squeeze()
imputed = df_imputed.loc[df_imputed.index.difference(df.index)].squeeze()

fig, axes = plt.subplots(2, figsize=(8, 4))

min_max = vaep.plotting.data.get_min_max_iterable([observed, imputed])

label_template = '{method} (N={n:,d})'
ax, _ = vaep.plotting.data.plot_histogram_intensities(
    observed,
    ax=axes[0],
    min_max=min_max,
    label=label_template.format(method='measured',
                                n=len(observed),
                                ),
    color='grey',
    alpha=1)
_ = ax.legend()
ax, _ = vaep.plotting.data.plot_histogram_intensities(
    imputed,
    ax=axes[1],
    min_max=min_max,
    label=label_template.format(method=f'{model_selected} imputed',
                                n=len(imputed),
                                ),
    color=color_model_mapping[model_selected],
    alpha=1)
_ = ax.legend()
