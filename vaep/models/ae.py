"""Autoencoder model trained using denoising procedure.

Variational Autencoder model adapter should be moved to vaep.models.vae.
Or model class could be put somewhere else.
"""
import logging
from typing import List, Union

import fastai.learner
import pandas as pd
import sklearn.pipeline
import torch
import torch.utils.data
from fastai.basics import L
from fastai.callback.core import Callback
from torch import nn

import vaep.io.dataloaders
import vaep.io.datasets
import vaep.io.datasplits
import vaep.models
import vaep.transform

from vaep.models import analysis

logger = logging.getLogger(__name__)


def get_preds_from_df(df: pd.DataFrame,
                      learn: fastai.learner.Learner,
                      transformer: vaep.transform.VaepPipeline,
                      position_pred_tuple: int = None,
                      dataset: torch.utils.data.Dataset = vaep.io.datasets.DatasetWithTarget):
    """Get predictions for specified DataFrame, using a fastai learner
    and a custom sklearn Pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to create predictions from.
    learn : fastai.learner.Learner
        fastai Learner with trained model
    transformer : vaep.transform.VaepPipeline
        Pipeline with separate encode and decode
    position_pred_tuple : int, optional
        In that the model returns multiple outputs, select the one which contains
        the predictions matching the target variable (VAE case), by default None
    dataset : torch.utils.data.Dataset, optional
        Dataset to build batches from, by default vaep.io.datasets.DatasetWithTarget

    Returns
    -------
    tuple
        tuple of pandas DataFrames (prediciton and target) based on learn.get_preds
    """
    dl = vaep.io.dataloaders.get_test_dl(df=df,
                                         transformer=transformer,
                                         dataset=dataset)
    res = learn.get_preds(dl=dl)  # -> dl could be int
    if position_pred_tuple is not None and issubclass(type(res[0]), tuple):
        res = (res[0][position_pred_tuple], *res[1:])
    res = L(res).map(lambda x: pd.DataFrame(
        x, index=df.index, columns=df.columns))
    res = L(res).map(lambda x: transformer.inverse_transform(x))
    return res


leaky_relu_default = nn.LeakyReLU(.1)


class Autoencoder(nn.Module):
    """Autoencoder base class.

    """

    def __init__(self,
                 n_features: int,
                 n_neurons: Union[int, List[int]],
                 activation=leaky_relu_default,
                 last_decoder_activation=None,
                 dim_latent: int = 10):
        """Initialize an Autoencoder

        Parameters
        ----------
        n_features : int
            Input dimension
        n_neurons : int
            Hidden layer dimension(s) in Encoder and Decoder.
            Can be a single integer or list.
        activation : [type], optional
            Activatoin for hidden layers, by default nn.Tanh
        last_encoder_activation : [type], optional
            Optional last encoder activation, by default nn.Tanh
        last_decoder_activation : [type], optional
            Optional last decoder activation, by default None
        dim_latent : int, optional
            Hidden space dimension, by default 10
        """
        super().__init__()
        self.n_features, self.n_neurons = n_features, list(L(n_neurons))
        self.layers = [n_features, *self.n_neurons]
        self.dim_latent = dim_latent

        # define architecture hidden layer
        def build_layer(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat),
                    nn.Dropout(0.2),
                    nn.BatchNorm1d(out_feat),
                    activation]

        # Encoder
        self.encoder = []

        for i in range(len(self.layers) - 1):
            in_feat, out_feat = self.layers[i:i + 2]
            self.encoder.extend(build_layer(in_feat=in_feat,
                                            out_feat=out_feat))
        self.encoder.append(nn.Linear(out_feat, dim_latent))

        self.encoder = nn.Sequential(*self.encoder)

        # Decoder
        self.layers_decoder = self.layers[::-1]
        assert self.layers_decoder is not self.layers
        assert out_feat == self.layers_decoder[0]

        self.decoder = build_layer(in_feat=self.dim_latent,
                                   out_feat=out_feat)

        i = -1  # in case a single hidden layer is passed
        for i in range(len(self.layers_decoder) - 2):
            in_feat, out_feat = self.layers_decoder[i:i + 2]
            self.decoder.extend(build_layer(in_feat=in_feat,
                                            out_feat=out_feat))
        in_feat, out_feat = self.layers_decoder[i + 1:i + 3]

        self.decoder.append(nn.Linear(in_feat, out_feat))
        if last_decoder_activation is not None:
            self.append(last_decoder_activation)
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_missing_values(df_train_wide: pd.DataFrame,
                       val_idx: pd.Index,
                       test_idx: pd.Index,
                       pred: pd.Series) -> pd.Series:
    """Build missing value predictions based on a set of prediction and splits.

    Parameters
    ----------
    df_train_wide : pd.DataFrame
        Training data in wide format.
    val_idx : pd.Index
        Indices (MultiIndex of Sample and Feature) of validation split
    test_idx : pd.Index
        Indices (MultiIndex of Sample and Feature) of test split
    pred : pd.Series
        Mulitindexed Series of all predictions.

    Returns
    -------
    pd.Series
        Multiindex series of missing values in training data which are not
        in validiation and test split.
    """
    # all idx missing in training data
    mask = df_train_wide.isna().stack()
    idx_real_na = mask.index[mask]
    # remove fake_na idx
    idx_real_na = (idx_real_na
                   .drop(val_idx)
                   .drop(test_idx))
    pred_real_na = pred.loc[idx_real_na]
    pred_real_na.name = 'intensity'
    return pred_real_na


class DatasetWithTargetAdapter(Callback):
    def before_batch(self):
        """Remove cont. values from batch (mask)"""
        mask, data = self.xb  # x_cat, x_cont (model could be adapted)
        self.learn._mask = mask != 1  # Dataset specific
        return data

    def after_pred(self):
        if len(self.yb):
            try:
                self.learn.yb = (self.y[self.learn._mask],)
            except IndexError:
                logger.warn(
                    f"Mismatch between mask ({self._mask.shape}) and y ({self.y.shape}).")
                # self.learn.y = None
                self.learn.yb = (self.xb[0],)
                self.learn.yb = (self.learn.xb[0].clone()[self._mask],)


class ModelAdapter(DatasetWithTargetAdapter):
    """Models forward only expects on input matrix.
    Apply mask from dataloader to both pred and targets.

    Keep original dimension, i.e. also predictions for NA."""

    def __init__(self, p=0.1):
        self.do = nn.Dropout(p=p)  # for denoising AE

    def before_batch(self):
        """Remove cont. values from batch (mask)"""
        data = super().before_batch()
        # dropout data using median
        if self.learn.training:
            self.learn.xb = (self.do(data),)
        else:
            self.learn.xb = (data,)

    def after_pred(self):
        self.learn._all_pred = self.pred.detach().clone()
        self.learn._all_y = None
        if len(self.yb):
            self.learn._all_y = self.y.detach().clone()
        super().after_pred()
        self.learn.pred = self.pred[self._mask]

    def after_loss(self):
        self.learn.pred = self.learn._all_pred
        if self._all_y is not None:
            self.learn.yb = (self._all_y,)


class ModelAdapterVAE(DatasetWithTargetAdapter):
    """Models forward method only expects one input matrix.
    Apply mask from dataloader to both pred and targets."""

    def before_batch(self):
        """Remove cont. values from batch (mask)"""
        data = super().before_batch()
        self.learn.xb = (data,)
        # data augmentation here?

    def after_pred(self):
        self.learn._all_pred = self.pred[0].detach().clone()
        self.learn._all_y = None
        if len(self.yb):
            self.learn._all_y = self.y.detach().clone()
        super().after_pred()
        if len(self.pred) == 3:
            pred, mu, logvar = self.pred  # return predictions
            self.learn.pred = (pred[self._mask], mu, logvar)  # is this flat?
        elif len(self.pred) == 4:
            x_mu, x_logvar, z_mu, z_logvar = self.pred
            self.learn.pred = (x_mu[self._mask], x_logvar[self._mask], z_mu, z_logvar)

    def after_loss(self):
        self.learn.pred = (self.learn._all_pred, *self.learn.pred[1:])
        if self._all_y is not None:
            self.learn.yb = (self._all_y,)


class AutoEncoderAnalysis(analysis.ModelAnalysis):

    def __init__(self,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,  # values to use for validation
                 model: torch.nn.modules.module.Module,
                 model_kwargs: dict,
                 transform: sklearn.pipeline.Pipeline,
                 decode: List[str],
                 bs=64
                 ):
        self.transform = vaep.transform.VaepPipeline(
            df_train=train_df,
            encode=transform,
            decode=decode)
        self.dls = vaep.io.dataloaders.get_dls(
            train_X=train_df,
            valid_X=val_df,
            transformer=self.transform, bs=bs)

        # M = data.train_X.shape[-1]
        self.kwargs_model = model_kwargs
        self.params = dict(self.kwargs_model)
        self.model = model(**self.kwargs_model)

        self.n_params_ae = vaep.models.calc_net_weight_count(self.model)
        self.params['n_parameters'] = self.n_params_ae
        self.learn = None

    def get_preds_from_df(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        if self.learn is None:
            raise ValueError("Assign Learner first as learn attribute.")
        return get_preds_from_df(df=df_wide, learn=self.learn, transformer=self.transform)

    def get_test_dl(self, df_wide: pd.DataFrame, bs: int = 64) -> pd.DataFrame:
        return vaep.io.dataloaders.get_test_dl(df=df_wide, transformer=self.transform, bs=bs)
