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
from vaep.sklearn.ae_transformer import AETransformer
import vaep.sampling
import numpy as np
import pandas as pd

from vaep.sklearn.cf_transformer import CollaborativeFilteringTransformer

fn_intensities: str = 'data/dev_datasets/HeLa_6070/protein_groups_wide_N50.csv'
df = pd.read_csv(fn_intensities, index_col=0)
df.head()

# %% [markdown]
# We will need the data in long format. Naming both the row and column index assures
# that the data can be transformed very easily into long format:

# %%
df.index.name = 'Sample ID'  # already set
df.columns.name = 'protein group'  # not set due to csv disk file format
df.head()

# %%
df = df.stack().to_frame('intensity')
df.head()

# %%
df = np.log2(df)
df.head()

# %% [markdown]
# The resulting DataFrame with one column has an `MulitIndex` with the sample and feature identifier.

# %%
# CollaborativeFilteringTransformer?

# %%
cf_model = CollaborativeFilteringTransformer(
    target_column='intensity',
    sample_column='Sample ID',
    item_column='protein group',
    out_folder='runs/scikit_interface')

# %%
cf_model.fit(df,
             cuda=True,
             epochs_max=5,
             )

# %%
df_imputed = cf_model.transform(df).unstack()
assert df_imputed.isna().sum().sum() == 0
df_imputed.head()


# %% [markdown]
# ## AutoEncoder architectures


# %%
# Reload data (for demonstration)

fn_intensities: str = 'data/dev_datasets/HeLa_6070/protein_groups_wide_N50.csv'
df = pd.read_csv(fn_intensities, index_col=0)
df.index.name = 'Sample ID'  # already set
df.columns.name = 'protein group'  # not set due to csv disk file format
df = np.log2(df)  # log transform
df.head()


# %%
freq_feat = df.notna().sum()
freq_feat.head()  # training data

# %%
val_X, train_X = vaep.sampling.sample_data(df.stack(),
                                           sample_index_to_drop=0,
                                           weights=freq_feat,
                                           frac=0.1,
                                           random_state=42,)
val_X, train_X = val_X.unstack(), train_X.unstack()
val_X = pd.DataFrame(pd.NA, index=train_X.index,
                     columns=train_X.columns).fillna(val_X)

# %%
val_X.shape, train_X.shape

# %%
train_X.notna().sum().sum(), val_X.notna().sum().sum(),

# %%
model = AETransformer(
    model='VAE',
    # model='DAE',
    hidden_layers=[512,],
    latent_dim=50,
    out_folder='runs/scikit_interface',
    batch_size=10,
)


# %%
model.fit(train_X, val_X,
          epochs_max=5,
          cuda=True)

# %%
df_imputed = model.transform(train_X)
df_imputed

# %%
# replace predicted values with val_X values
df_imputed = df_imputed.replace(val_X)

# %%
