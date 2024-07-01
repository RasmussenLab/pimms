"""Scikit-learn style interface for Denoising and Variational Autoencoder model."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import sklearn
from fastai import learner
from fastai.basics import *
from fastai.callback.all import *
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.learner import Learner
from fastai.losses import MSELossFlat
from fastai.torch_basics import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

import vaep.models as models
# patch plotting function
from vaep.models import ae, plot_loss

learner.Recorder.plot_loss = plot_loss


default_pipeline = sklearn.pipeline.Pipeline(
    [
        ('normalize', StandardScaler()),
        ('impute', SimpleImputer(add_indicator=False))
    ])


class AETransformer(TransformerMixin, BaseEstimator):
    """Autoencoder transformer (Denoising or Variational).

    Autoencoder transformer which can be used to impute missing values
    in a dataset it is fitted to. The data is standard normalized
    for fitting the model, but imputations are provided on the original scale
    after internally fitting the model.

    The data uses the wide data format with samples as rows and features as columns.


    Parameters
    ----------
    hidden_layers : list[int]
        Architecture for encoder. Decoder is mirrored.
    dim_latent : int, optional
        Hidden space dimension, by default 15
    out_folder : str, optional
        Output folder for model, by default '.'
    model : str, optional
        Model type ("VAE", "DAE"), by default 'VAE'
    batch_size : int, optional
        Batch size for training, by default 64

    """

    def __init__(self,
                 hidden_layers: list[int],
                 latent_dim: int = 15,
                 out_folder: str = '.',
                 model='VAE',
                 #  y_range:Optional[tuple[int]]=None,
                 batch_size: int = 64,
                 ):
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.out_folder = Path(out_folder)
        self.out_folder.mkdir(exist_ok=True, parents=True)

        if model == 'VAE':
            self.model = models.vae.VAE
            self.cbs = [ae.ModelAdapterVAE()]
            self.loss_fct = models.vae.loss_fct
        elif model == 'DAE':
            self.model = ae.Autoencoder
            self.cbs = [ae.ModelAdapter(p=0.2)]
            self.loss_fct = MSELossFlat(reduction='sum')
        else:
            raise ValueError(f'Unknown model {model}, choose either "VAE" or "DAE"')
        self.model_name = model

    def fit(self,
            X: pd.DataFrame,
            y: pd.DataFrame = None,
            epochs_max: int = 100,
            cuda: bool = True,
            patience: Optional[int] = None):
        """Fit the model to the data.

        Parameters
        ----------
        X : pd.DataFrame
            training data of dimension N_samples x M_features
        y : pd.DataFrame, optional
            validation data points which are missing in X of dimension
            N_sample x M_features, by default None
        epochs_max : int, optional
            Maximal number of epochs to train, by default 100
        cuda : bool, optional
            If the model should be trained with an accelerator, by default True
        patience : Optional[int], optional
            If added, early stopping is added with specified patience, by default None

        Returns
        -------
        AETransformer
            Return itself fitted to the training data.
        """
        self.analysis = ae.AutoEncoderAnalysis(  # datasplits=data,
            train_df=X,
            val_df=y,
            model=self.model,
            model_kwargs=dict(n_features=X.shape[-1],
                              n_neurons=self.hidden_layers,
                              last_decoder_activation=None,
                              dim_latent=self.latent_dim),
            transform=default_pipeline,
            decode=['normalize'],
            bs=self.batch_size)

        self.n_params = self.analysis.n_params_ae
        if cuda:
            self.analysis.model = self.analysis.model.cuda()

        cbs = self.cbs
        if patience is not None:
            cbs = [*self.cbs, EarlyStoppingCallback(patience=patience)]
        self.analysis.learn = Learner(dls=self.analysis.dls,
                                      model=self.analysis.model,
                                      loss_func=self.loss_fct,
                                      cbs=cbs
                                      )

        suggested_lr = self.analysis.learn.lr_find()
        self.analysis.params['suggested_inital_lr'] = suggested_lr.valley
        self.analysis.learn.fit_one_cycle(epochs_max, lr_max=suggested_lr.valley)
        self.epochs_trained_ = self.analysis.learn.epoch + 1
        N_train_notna = X.notna().sum().sum()
        N_val_notna = None
        if y is not None:
            N_val_notna = y.notna().sum().sum()
        self.fig_loss_ = models.plot_training_losses(
            self.analysis.learn, self.model_name,
            folder=self.out_folder,
            norm_factors=[N_train_notna, N_val_notna])
        return self

    def transform(self, X):
        """Impute the data using the trained model.

        Parameters
        ----------
        X : pd.DataFrame
            The data to be imputed, shape (N_samples, N_features).


        Returns
        -------
        X_transformed : array, shape (N_samples, M_features)
            Return the imputed DataFrame using the model.
        """
        # Check is fit had been called
        check_is_fitted(self, 'epochs_trained_')

        self.analysis.model.eval()

        pred, target = ae.get_preds_from_df(
            df=X,
            learn=self.analysis.learn,
            position_pred_tuple=0,
            transformer=self.analysis.transform)
        return X.fillna(pred)
