"""Scikit-learn style interface for Denoising and Variational Autoencoder model."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sklearn

from pathlib import Path
import pandas as pd

from fastai.losses import MSELossFlat
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.learner import Learner
from fastai.basics import *
from fastai.callback.all import *
from fastai.torch_basics import *

from fastai import learner

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Optional

import vaep.models as models
from vaep.models import ae


# patch plotting function
from vaep.models import plot_loss
learner.Recorder.plot_loss = plot_loss


default_pipeline = sklearn.pipeline.Pipeline(
    [
        ('normalize', StandardScaler()),
        ('impute', SimpleImputer(add_indicator=False))
    ])


class AETransformer(TransformerMixin, BaseEstimator):
    """Collaborative Filtering transformer.


    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
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
        # ! patience?
        # EarlyStoppingCallback(patience=args.patience)

    def fit(self,
            X: pd.DataFrame,
            y: pd.DataFrame = None,
            epochs_max: int = 100,
            cuda: bool = True,
            patience: Optional[int] = None):
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

        # results = []
        # loss_fct = partial(models.vae.loss_fct, results=results)
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
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
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
