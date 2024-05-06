"""Scikit-learn style interface for Collaborative Filtering model."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fastai import learner
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.collab import *
from fastai.collab import CollabDataLoaders, EmbeddingDotBias, TabularCollab
from fastai.data.block import TransformBlock
from fastai.data.transforms import IndexSplitter
from fastai.learner import Learner
from fastai.losses import MSELossFlat
from fastai.tabular.all import *
from fastai.tabular.core import Categorify
from fastai.torch_core import default_device
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import vaep
import vaep.models as models
# patch plotting function
from vaep.models import collab, plot_loss

learner.Recorder.plot_loss = plot_loss


class CollaborativeFilteringTransformer(TransformerMixin, BaseEstimator):
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
                 target_column: str,
                 sample_column: str,
                 item_column: str,
                 n_factors: int = 15,
                 out_folder: str = '.',
                 #  y_range:Optional[tuple[int]]=None,
                 batch_size: int = 4096,
                 ):
        self.target_column = target_column
        self.item_column = item_column
        self.sample_column = sample_column
        self.n_factors = n_factors
        self.out_folder = Path(out_folder)
        self.out_folder.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size

    def fit(self, X: pd.Series, y: pd.Series = None,
            cuda: bool = True,
            patience: int = 1,
            epochs_max=20,):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        # ! X = check_array(X, accept_sparse=True)
        self.model_kwargs = dict(


            n_factors=self.n_factors,
            y_range=(int(X.squeeze().min()),
                     int(X.squeeze().max()) + 1)
        )
        if not cuda:
            default_device(use=False)  # set to cpu
        if y is not None:
            X, frac = collab.combine_data(X, y)
        else:
            X, frac = X.reset_index(), 0.0

        self.dls = CollabDataLoaders.from_df(
            X,
            valid_pct=frac,
            seed=42,
            user_name=self.sample_column,
            item_name=self.item_column,
            rating_name=self.target_column,
            bs=self.batch_size)
        splits = None
        if y is not None:
            idx_splitter = IndexSplitter(list(range(len(X), len(X) + len(y))))
            splits = idx_splitter(X)

        self.cat_names = [self.sample_column, self.item_column]
        self.to = TabularCollab(df=X,
                                procs=[Categorify],
                                cat_names=self.cat_names,
                                y_names=[self.target_column],
                                y_block=TransformBlock(),
                                splits=splits)
        self.dls = self.to.dataloaders(path='.', bs=self.batch_size)

        self.model = EmbeddingDotBias.from_classes(
            classes=self.dls.classes,
            **self.model_kwargs)

        self.n_params = models.calc_net_weight_count(self.model)
        # ana_collab.params['n_parameters'] = args.n_params
        self.learn = Learner(dls=self.dls,
                             model=self.model,
                             loss_func=MSELossFlat(),
                             cbs=EarlyStoppingCallback(patience=patience) if y is not None else None,
                             model_dir=self.out_folder)
        if cuda:
            self.learn.model = self.learn.model.cuda()

        #
        suggested_lr = self.learn.lr_find()
        print(f"{suggested_lr.valley = :.5f}")

        self.learn.fit_one_cycle(epochs_max,

                                 lr_max=suggested_lr.valley)
        self.plot_loss(y)
        self.epochs_trained_ = self.learn.epoch + 1
        self.model_kwargs['suggested_inital_lr'] = suggested_lr.valley
        # ? own method?
        # self.learn.save('collab_model')

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

        # ! Input validation
        # X = check_array(X, accept_sparse=True)

        X = X.squeeze()
        mask = X.unstack().isna().stack()
        idx_na = mask.loc[mask].index
        dl_real_na = self.dls.test_dl(idx_na.to_frame())
        pred_na, _ = self.learn.get_preds(dl=dl_real_na)
        pred_na = pd.Series(pred_na, idx_na, name=self.target_column)
        return pd.concat([X, pred_na])

    def plot_loss(self, y, figsize=(8, 4)):  # -> Axes:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title('CF loss: Reconstruction loss')
        self.learn.recorder.plot_loss(skip_start=5, ax=ax,
                                      with_valid=True if y is not None else False)
        vaep.savefig(fig, name='collab_training',
                     folder=self.out_folder)
        self.model_kwargs['batch_size'] = self.batch_size
        vaep.io.dump_json(self.model_kwargs,
                          self.out_folder / 'model_params_{}.json'.format('CF'))
        return ax
