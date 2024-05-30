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

# %%
import os
IN_COLAB = 'COLAB_GPU' in os.environ

fn_intensities = 'data/dev_datasets/HeLa_6070/protein_groups_wide_N50.csv'
if IN_COLAB:
    fn_intensities = 'https://raw.githubusercontent.com/RasmussenLab/pimms/main/project/data/dev_datasets/HeLa_6070/protein_groups_wide_N50.csv'

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from vaep.plotting.defaults import color_model_mapping
import vaep.plotting.data
import vaep.sampling

from vaep.sklearn.cf_transformer import CollaborativeFilteringTransformer
from vaep.sklearn.ae_transformer import AETransformer

vaep.plotting.make_large_descriptors(8)

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
df.index.name = 'Sample ID'  # already set
df.columns.name = 'protein group'  # not set due to csv disk file format
df.head()

# %% [markdown]
#

# %% [markdown]
# Transform the data using the logarithm, here using base 2:

# %%
df = np.log2(df + 1)
df.head()


# %% [markdown]
# two plots on data availability:
#
# 1. proportion of missing values per feature median (N = protein groups)
# 2. CDF of available intensities per protein group

# %%
ax = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=df, type='boxplot')


# %%
df.notna().sum().sort_values().plot()


# %% [markdown]
# define a minimum feature and sample frequency for a feature to be included

# %%
SELECT_FEAT = True


def select_features(df, feat_prevalence=.2, axis=0):
    # # ! vaep.filter.select_features
    N = df.shape[axis]
    minimum_freq = N * feat_prevalence
    freq = df.notna().sum(axis=axis)
    mask = freq >= minimum_freq
    print(f"Drop {(~mask).sum()} along axis {axis}.")
    freq = freq.loc[mask]
    if axis == 0:
        df = df.loc[:, mask]
    else:
        df = df.loc[mask]
    return df


if SELECT_FEAT:
    # potentially this can take a few iterations to stabilize.
    df = select_features(df, feat_prevalence=.2)
    df = select_features(df=df, feat_prevalence=.3, axis=1)
df.shape


# %% [markdown]
# Transform to long-data format:

# %%
df = df.stack().to_frame('intensity')
df.head()


# %% [markdown]
# The resulting DataFrame with one column has an `MulitIndex` with the sample and feature identifier.

# %% [markdown]
# ## Collaborative Filtering

# %%
# # # # CollaborativeFilteringTransformer?

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
cf_model.fit(df,
             cuda=False,
             epochs_max=20,
             )

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

# %%
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
# Reload data (for demonstration)

df = pd.read_csv(fn_intensities, index_col=0)
df.index.name = 'Sample ID'  # already set
df.columns.name = 'protein group'  # not set due to csv disk file format
df = np.log2(df + 1)  # log transform
df.head()

# %% [markdown]
# The AutoEncoder model currently need validation data for training.
# We will use 10% of the training data for validation.
# > Expect this limitation to be dropped in the next release. It will still be recommended
# > to use validation data for early stopping.

# %%
freq_feat = df.notna().sum()
freq_feat.head()  # training data

# %% [markdown]
# We will use the `sampling` module to sample the validation data from the training data.
# Could be split differently by providing another `weights` vector.

# %%
val_X, train_X = vaep.sampling.sample_data(df.stack(),
                                           sample_index_to_drop=0,
                                           weights=freq_feat,
                                           frac=0.1,
                                           random_state=42,)
val_X, train_X = val_X.unstack(), train_X.unstack()
val_X = pd.DataFrame(pd.NA, index=train_X.index,
                     columns=train_X.columns).fillna(val_X)

# %% [markdown]
# Training data and validation data have the same shape:

# %%
val_X.shape, train_X.shape

# %% [markdown]
# ... but different number of intensities (non-missing values):

# %%
train_X.notna().sum().sum(), val_X.notna().sum().sum(),

# %% [markdown]
# Select either `DAE` or `VAE` model:

# %%
model_selected = 'VAE'  # 'DAE'
model = AETransformer(
    model=model_selected,
    hidden_layers=[512,],
    latent_dim=50,
    out_folder='runs/scikit_interface',
    batch_size=10,
)

# %%
model.fit(train_X, val_X,
          epochs_max=50,
          cuda=False)

# %%
df_imputed = model.transform(train_X)
df_imputed

# %% [markdown]
# Evaluate the model using the validation data:

# %%
pred_val = val_X.stack().to_frame('observed')
pred_val[model_selected] = df_imputed.stack()
pred_val

# %%
val_metrics = vaep.models.calculte_metrics(pred_val, 'observed')
# val_metrics = metrics.add_metrics(
#     pred_val, key='test data')
# val_metrics = pd.DataFrame(val_metrics)
# val_metrics
pd.DataFrame(val_metrics)

# %%
fig, ax = plt.subplots(figsize=(8, 2))

ax, errors_binned = vaep.plotting.errors.plot_errors_by_median(
    pred=pred_val,
    target_col='observed',
    feat_medians=train_X.median(),
    ax=ax,
    metric_name='MAE',
    palette=color_model_mapping
)

# %% [markdown]
# replace predicted values with validation data values

# %%
df_imputed = df_imputed.replace(val_X)

# %%
df = df.stack()  # long-format
df_imputed = df_imputed.stack()  # long-format
observed = df_imputed.loc[df.index]
imputed = df_imputed.loc[df_imputed.index.difference(df.index)]

# %%
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
    label=label_template.format(method=f'{model_selected} imputed',
                                n=len(imputed),
                                ),
    color=color_model_mapping[model_selected],
    alpha=1)
_ = ax.legend()


# %%
