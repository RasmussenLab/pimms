"""Scikit-learn style interface for Collaborative Filtering model."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fastai import learner
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.collab import *
from fastai.collab import EmbeddingDotBias, TabularCollab
from fastai.data.block import TransformBlock
from fastai.data.transforms import IndexSplitter
from fastai.learner import Learner
from fastai.losses import MSELossFlat
from fastai.tabular.all import *
from fastai.tabular.all import TransformBlock
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

    Collaborative filtering operates on long data specifying two identifiers
    (sample and feature) with a quantitative value to predict. Therefore we need to specify
    three columns. The sample and feature identifiers are embedded into a space which is
    then used to predict the quantitative value.

    The data is expected as a Series with a MultiIndex of the sample and feature identifiers,
    and the quantitative value as its values.

    Parameters
    ----------
    target_column : str
        Target column name to predict, e.g. intensity
    item_column: str
        Column name for features (items) to embed, e.g. peptides
    sample_column: str
        Sample column name, e.g. Sample_ID
    n_factors : int, optional
        number of dimension of item and sample embeddings, by default 15
    out_folder : str, optional
        Output folder for model, by default '.'
    batch_size : int, optional
        Batch size for training of data in long format, by default 4096

    """

    def __init__(self,
                 target_column: str,
                 sample_column: str,
                 item_column: str,
                 n_factors: int = 15,
                 out_folder: str = '.',
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
            epochs_max=20,
            cuda: bool = True,
            patience: int = 1):
        """Fit the collaborative filtering model to the data provided in long-format.

        Parameters
        ----------
        X : Series, shape (n_values, )
            The training data as a Series with the target_column as it values
            and target_column as its name. The Series has a MultiIndex defined by the
            item_column and sample_column.
            Is of shape (n_values, )

        y : Series, optional
            The validation data as a Series with the target_column as it values
            and target_column as its name. The Series has a MultiIndex defined by the
            item_column and sample_column.
            Is of shape (n_values, ), by default None

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
        self.model_kwargs = dict(


            n_factors=self.n_factors,
            y_range=(int(X.squeeze().min()),
                     int(X.squeeze().max()) + 1)
        )
        if not cuda:
            default_device(use=False)  # set to cpu
        if y is not None:
            # Concatenate train and validation observations into on dataframe
            first_N_train = len(X)
            X, _ = collab.combine_data(X, y)
        else:
            X, _ = X.reset_index(), 0.0

        splits = None
        if y is not None:
            # specify positional indices of validation data
            idx_splitter = IndexSplitter(list(range(first_N_train, len(X))))
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
        self.learn = Learner(dls=self.dls,
                             model=self.model,
                             loss_func=MSELossFlat(),
                             cbs=EarlyStoppingCallback(patience=patience) if y is not None else None,
                             model_dir=self.out_folder)
        if cuda:
            self.learn.model = self.learn.model.cuda()

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
        """Predict the mising features in the long data based on the index of
        sample_column and item_column.

        Parameters
        ----------
        X : Series, shape (n_samples, )
            The training data with columns target_column, item_column and sample_column.

        Returns
        -------
        X_transformed : pd.Series (n_samples, n_features)
            The complete data with imputed values in long format
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
        """Plot the training and validation loss of the model."""
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
