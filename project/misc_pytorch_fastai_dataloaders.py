# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # `DataLoaders` for feeding data into models

# %%
import numpy as np
import pandas as pd

import fastai
from fastai.tabular.core import Normalize
from fastai.tabular.core import FillMissing
from fastai.tabular.core import TabularPandas
from fastai.tabular.core import IndexSplitter
# make DataLoaders.test_dl work for DataFrames as test_items:

# from fastai.tabular.all import *
from fastai.tabular.all import TabularDataLoaders
from fastcore.transform import Pipeline

import torch

from vaep.logging import setup_nb_logger
setup_nb_logger()

from vaep.io.datasplits import DataSplits
from vaep.io.datasets import DatasetWithMaskAndNoTarget, to_tensor
from vaep.transform import VaepPipeline
from vaep.models import ae
from vaep.utils import create_random_df

np.random.seed(42)
print(f"fastai version: {fastai.__version__}")
print(f"torch  version: {torch.__version__}")

# %%
from fastcore.transform import Pipeline

from fastcore.basics import store_attr
class FillMissingKeepAll(FillMissing):
    """Replacement for `FillMissing` including also non-missing features
    in the training data which might be missing in the validation or test data.
    """
    def setups(self, to):
        store_attr(but='to', na_dict={n:self.fill_strategy(to[n], self.fill_vals[n])
                            for n in to.conts.keys()})
        self.fill_strategy = self.fill_strategy.__name__



# %% [markdown]
# Create data
#
# - train data without missings
# - validation and test data with missings
#
# Could be adapted to have more or less missing in training, validation or test data. Choosen as in current version the validation data cannot contain features with missing values which were not missing in the training data.

# %%
N, M = 150, 15

create_df = create_random_df

X = create_df(N, M)
X = X.append(create_df(int(N*0.3), M, prop_na=.1, start_idx=len(X)))

idx_val = X.index[N:] # RandomSplitter could be used, but used to show IndexSplitter usage with Tabular

X_test = create_df(int(N*0.1), M, prop_na=.1, start_idx=len(X))

data = DataSplits(train_X=X.loc[X.index.difference(idx_val)],
                  val_y=X.loc[idx_val],
                  test_y=X_test,
                  is_wide_format=True)

data.val_y.loc[data.val_y.isna().any(axis=1), data.val_y.isna().any(axis=0)]

# %% [markdown]
# ## Collab

# %%

# %% [markdown]
# ## Denoising Autoencoder

# %% [markdown]
# ### DataSet `Tabular`
#
# - `fastai.tabular.core.Tabular`
#
#
# Adding procs / transforms manually
#
# ```python
# cont_names = list(splits.train_X.columns)
# to = TabularPandas(splits.train_X, cont_names=cont_names, do_setup=False)
#
# tf_norm = NORMALIZER()
# tf_fillna = FillMissing(add_col=True)
#
# _ = tf_norm.setups(to)  # returns to
# _ = tf_fillna.setup(to)
# ```
#
# No added in a manuel pipeline. See [opened issue](https://github.com/fastai/fastai/issues/3530) on `Tabular` behaviour.
# Setting transformation (procs) in the constructor is somehow not persistent, although very similar code is called.
#
# ```
# # not entirely empty, but to.procs.fs needs to be populated
# type(to.procs), to.procs.fs # __call__, setup, decode, fs
# ```

# %%
X = data.train_X.append(data.val_y)

splits = X.index.get_indexer(data.val_y.index) # In Tabular iloc is used, not loc for splitting
splits = IndexSplitter(splits)(X) # splits is are to list of integer indicies (for iloc)
        
procs = [Normalize, FillMissingKeepAll]

to = TabularPandas(X, procs=procs, cont_names=X.columns.to_list(), splits=splits) # to = tabular object

print("Tabular object:", type(to))
to.items.head()

# %% [markdown]
# Test data with procs

# %%
procs = to.procs
procs.fs

# %% [markdown]
# Let's format this to see what it does
#
# ```python
# # (#2)
# [
# FillMissingKeepAll -- 
# {'fill_strategy': <function FillStrategy.median at 0x0000023845497E50>, 
#  'add_col': True, 
#  'fill_vals': defaultdict(<class 'int'>,  {'feat_00': 0, 'feat_01': 0, 'feat_02': 0, ..., 'feat_14': 13.972452}
# }:
#     encodes: (object,object) -> encodes
#     decodes: ,
# Normalize -- 
# {'mean': None, 'std': None, 'axes': (0, 2, 3),
#  'means': {'feat_00': 14.982738, 'feat_01': 13.158741, 'feat_02': 14.800485, ..., 'feat_14': 8.372757}
# }:
#     encodes: (TensorImage,object) -> encodes
#              (Tabular,object) -> encodes
#     decodes: (TensorImage,object) -> decodes
#              (Tabular,object) -> decodes
# ]
#
# ```

# %%
procs

# %%
# Check behaviour
procs.encodes

# %% [markdown]
# #### DataLoader

# %%
dls = to.dataloaders(bs=4)
dls.show_batch()

# %%
dls.one_batch()

# %%
[x.dtype for x in dls.one_batch()]

# %% [markdown]
# #### transfrom test data using `DataLoaders.test_dl`

# %%
# test_ds = TabularPandas(data.test_y, cont_names=data.test_y.columns.to_list())
dl_test = dls.test_dl(data.test_y.copy())
dl_test.xs.head()

# %%
dl_test.show_batch()

# %% [markdown]
# #### Transform test data manuelly

# %%
to_test = TabularPandas(data.test_y.copy(), procs=None, cont_names=data.test_y.columns.to_list(), splits=None, do_setup=True)
_ = procs(to_test) # inplace operation
to_test.items.head()

# %%
data.test_y.head()

# %% [markdown]
# #### Feeding one batch to the model

# %%
cats, conts, ys =  dls.one_batch()

# %%
model = ae.Autoencoder(n_features=M, n_neurons=int(
    M/2), last_decoder_activation=None, dim_latent=10)
model

# %% [markdown]
# The forward pass just uses the conts features

# %%
model(conts)

# %% [markdown]
# #### target
# - missing puzzle piece is to have a `callable` y-block which transforms part of the input. In principle it could be the same as the continous features

# %% [markdown]
# ### PyTorch Dataset

# %%
train_ds = DatasetWithMaskAndNoTarget(df=data.train_X)
valid_ds = DatasetWithMaskAndNoTarget(df=data.val_y)
train_ds[-1]

# %% [markdown]
# #### DataLoaders

# %%
from fastai.data.core import DataLoaders

dls = DataLoaders.from_dsets(train_ds, valid_ds,
                             bs=4)

dls.valid.one_batch()

# %% [markdown]
# #### DataLoaders with Normalization fastai Transform

# %%
from fastai.tabular.all import * 
class Normalize(Transform):
    def setup(self, array):
        self.mean = array.mean()  # this assumes tensor, numpy arrays and alike
        # should be applied along axis 0 (over the samples)
        self.std = array.std()  # ddof=0 in scikit-learn
    
    def encodes(self, x): # -> torch.Tensor: # with type annotation this throws an error
        x_enc = (x - self.mean) / self.std
        return x_enc

    def decodes(self, x_enc:torch.tensor) -> torch.Tensor:
        x = (self.std * x_enc) + self.mean
        return x
    
o_tf_norm = Normalize()
o_tf_norm.setup(data.train_X)
o_tf_norm(data.val_y.head()) # apply this manueally to each dataset

# %%
o_tf_norm.encodes # object= everything

# %%
train_ds = DatasetWithMaskAndNoTarget(df=o_tf_norm(data.train_X))
valid_ds = DatasetWithMaskAndNoTarget(df=o_tf_norm(data.val_y))

dls = DataLoaders.from_dsets(
    train_ds,
    valid_ds,
    #  tfms=[o_tf_norm],
    #  after_batch=[o_tf_norm],
    bs=4)

dls.valid.one_batch()

# %%
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_less

assert (dls.valid.one_batch()[1] < 0.0).any(), "Normalization did not work."
with pytest.raises(AttributeError):
    DatasetWithMaskAndNoTarget(df=data.val_y, transformer=o_tf_norm)
    
# assert_array_almost_equal(DatasetWithMaskAndNoTarget(df=data.val_y, transformer=o_tf_norm)[0][1], DatasetWithMaskAndNoTarget(df=o_tf_norm(data.val_y))[0][1])
# with pytest.raises(AttributeError):
#        valid_ds.inverse_transform(dls.valid.one_batch()[1])

# %% [markdown]
# #### DataLoaders with Normalization sklearn transform
#
# - solve transformation problem by composition
# - inverse transform only used for 

# %%
import sklearn
# from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import vaep
# import importlib; importlib.reload(vaep); importlib.reload(vaep.transform)

dae_default_pipeline = sklearn.pipeline.Pipeline(
    [
        ('normalize', StandardScaler()),
        ('impute', SimpleImputer(add_indicator=False))
    ])
# new procs, transform equal encode, inverse_transform equals decode
dae_transforms = VaepPipeline(
    df_train=data.train_X, encode=dae_default_pipeline, decode=['normalize'])

# %%
valid_ds = DatasetWithMaskAndNoTarget(data.val_y, dae_transforms)
valid_ds[:4]

# %%
from vaep.io.dataloaders import get_dls
dls = get_dls(data.train_X, data.val_y, dae_transforms, bs=4)    
dls.valid.one_batch()

# %%
test_dl = DataLoader(
    dataset=DatasetWithMaskAndNoTarget(data.test_y, dae_transforms),
    shuffle=False,
    bs=4)
test_dl.one_batch()

# %%
dae_transforms.inverse_transform(test_dl.one_batch()[1]) # here the missings are not replaced

# %%
data.test_y.head(4)

# %% [markdown]
# ### FastAi Transfrom (as Dataset)
#
# - adding `Transforms` not possible, I openend a [discussion](https://forums.fast.ai/t/correct-output-type-for-tensor-created-from-dataframe-custom-new-task-tutorial/92564)

# %%
from typing import Tuple
from fastai.tabular.all import *
# from fastai.torch_core import TensorBase


class DatasetTransform(Transform):
    def __init__(self, df: pd.DataFrame):
        if not issubclass(type(df), pd.DataFrame):
            raise ValueError(
                f'please pass a pandas DataFrame, not: {type(df) = }')
        self.mask_obs = df.isna()  # .astype('uint8') # in case 0,1 is preferred
        self.data = df

    def encodes(self, idx): # -> Tuple[torch.Tensor, torch.Tensor]: # annotation is interpreted
        mask = self.mask_obs.iloc[idx]
        data = self.data.iloc[idx]
        # return (self.to_tensor(mask), self.to_tensor(data))
        # return (Tensor(mask), Tensor(data))
        return (tensor(data), tensor(mask)) #TabData, TabMask

    def to_tensor(self, s: pd.Series) -> torch.Tensor:
        return torch.from_numpy(s.values)


train_tl = TfmdLists(
    range(len(data.train_X)),
    DatasetTransform(data.train_X))
valid_tl = TfmdLists(
    range(len(data.val_y)),
    DatasetTransform(data.val_y))

dls = DataLoaders.from_dsets(train_tl, valid_tl,
#                              after_item=[Normalize],
#                              after_batch=[Normalize],
                             bs=4)
print(f"\n{DatasetTransform.encodes = }")
dls.one_batch()

# %% [markdown]
# ## Variational Autoencoder

# %%
from vaep.transform import MinMaxScaler

args_vae = {}
args_vae['SCALER'] = MinMaxScaler
# select initial data: transformed vs not log transformed
scaler = args_vae['SCALER']().fit(data.train_X)

_transform_fct = scaler.transform

train_ds = DatasetWithMaskAndNoTarget(df=_transform_fct(data.train_X))
valid_ds = DatasetWithMaskAndNoTarget(df=_transform_fct(data.val_y))

dls = DataLoaders.from_dsets(train_ds, valid_ds,
                             bs=4)
dls.one_batch()

# %% [markdown]
# ## FastAi version

# %%
