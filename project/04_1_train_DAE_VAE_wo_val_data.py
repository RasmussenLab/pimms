# %% [markdown]
# # Scikit-learn styple transformers of the data
#
# 1. Load data into pandas dataframe
# 2. Fit transformer on training data
# 3. Impute only missing values with predictions from model
#
# Autoencoders need wide training data, i.e. a sample with all its features' intensities, whereas
# Collaborative Filtering needs long training data, i.e. sample identifier a feature identifier and the intensity.
# Both data formats can be transformed into each other, but models using long data format do not need to
# take care of missing values.

# %%
import os
import pandas as pd
import numpy as np

import vaep.plotting.data
from vaep.sklearn.ae_transformer import AETransformer
import vaep.sampling


IN_COLAB = 'COLAB_GPU' in os.environ

fn_intensities = 'data/dev_datasets/HeLa_6070/protein_groups_wide_N50.csv'
if IN_COLAB:
    fn_intensities = 'https://raw.githubusercontent.com/RasmussenLab/pimms/main/project/data/dev_datasets/HeLa_6070/protein_groups_wide_N50.csv'

# %%


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
# ## AutoEncoder architectures

# %%
# Reload data (for demonstration)

df = pd.read_csv(fn_intensities, index_col=0)
df.index.name = 'Sample ID'  # already set
df.columns.name = 'protein group'  # not set due to csv disk file format
df = np.log2(df + 1)  # log transform
df.head()

# %% [markdown]
# Test `DAE` or `VAE` model without validation data:

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
model.fit(df,
          epochs_max=2,
          cuda=False)

# %%
df_imputed = model.transform(df)
df_imputed

# %% [markdown]
# DAE

# %%
model_selected = 'DAE'
model = AETransformer(
    model=model_selected,
    hidden_layers=[512,],
    latent_dim=50,
    out_folder='runs/scikit_interface',
    batch_size=10,
)

# %%
model.fit(df,
          epochs_max=2,
          cuda=False)

# %%
df_imputed = model.transform(df)
df_imputed
