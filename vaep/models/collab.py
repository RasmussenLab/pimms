import logging
from typing import Tuple, List

import pandas as pd
import torch

import fastai
# from fastai.collab import Module, Embedding, sigmoid_range
# from fastai.collab import EmbeddingDotBias
from fastai.tabular.all import *
from fastai.collab import *

from . import analysis
import vaep.io.datasplits
import vaep.io.dataloaders


logger = logging.getLogger(__name__)


class DotProductBias(Module):
    """Explicit implementation of fastai.collab.EmbeddingDotBias."""

    def __init__(self, n_samples, n_peptides, dim_latent_factors, y_range=(14, 30)):
        self.sample_factors = Embedding(n_samples, dim_latent_factors)
        self.sample_bias = Embedding(n_samples, 1)
        self.peptide_factors = Embedding(
            n_peptides, dim_latent_factors)
        self.peptide_bias = Embedding(n_peptides, 1)
        self.y_range = y_range

    def forward(self, x):
        samples = self.sample_factors(x[:, 0])
        peptides = self.peptide_factors(x[:, 1])
        res = (samples * peptides).sum(dim=1, keepdim=True)
        res += self.sample_bias(x[:, 0]) + self.peptide_bias(x[:, 1])
        return sigmoid_range(res, *self.y_range)


def combine_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Helper function to combine training and validation data in long-format. The 
    training and validation data will be mixed up in CF training as the sample
    embeddings have to be trained for all samples. The returned frac can be used to have
    the same number of (non-missing) validation samples as before.

    Parameters
    ----------
    train_df : pd.DataFrame
        Consecutive training data in long-format, each row having (unit, feature, value)
    val_df : pd.DataFrame
        Consecutive training data in long-format, each row having (unit, feature, value)

    Returns
    -------
    Tuple[pd.DataFrame, List[list, list]]
        Pandas DataFrame of concatenated samples of training and validation data.
        Fraction of samples originally in validation data.
    """
    X = train_df.append(val_df).reset_index()
    frac = len(val_df) / (len(train_df)+len(val_df))
    return X, frac


def collab_dot_product(sample_embeddings: torch.tensor, sample_bias: torch.tensor,
                       feat_embeddings: Embedding, feat_bias: Embedding, items: torch.tensor,
                       y_range=None) -> torch.tensor:
    """CF dot product for a single sample using a set of feature items.

    Parameters
    ----------
    sample_embeddings : torch.tensor
        Sample emedding to use prediction for
    sample_bias : torch.tensor
        sample bias
    feat_embeddings : Embedding
        Embedding layer for fetch feature items from
    feat_bias : Embedding
        Embedding layer to fetch feature biases from
    items : torch.tensor
        Set of feature IDs to use
    y_range : tuple, optional
        range transformation based on sigmoid function, by default None

    Returns
    -------
    torch.tensor
        target predictions for sample features.

    Example
    -------
    # learn is a CF Learner
    idx = learn.classes['Sample ID'].map_objs(['sample1', 'sample2'])
    idx = torch.tensor(idx)
    collab_dot_product(learn.u_weight(idx), learn.u_bias(idx),
                   learn.i_weight, learn.i_bias, idx_feat, # this is abritrary
                   y_range=learn.y_range)
    """
    dot = sample_embeddings * feat_embeddings(items)
    res = dot.sum(1) + sample_bias.squeeze() + feat_bias(items).squeeze()
    res = res.detach()
    if y_range is None:
        return res
    return torch.sigmoid(res) * (y_range[1]-y_range[0]) + y_range[0]


def collab_prediction(idx_samples: torch.tensor,
                      learn: fastai.learner.Learner,
                      index_samples: pd.Index = None) -> pd.DataFrame:
    """Based on a CF model Learner, calculate all out of sample predicitons
    for all features trained.

    Parameters
    ----------
    idx_samples : torch.tensor
        An array containing the neighreast neighbors in the training data for 
        set of list of test samples. Normallay obtained from a sklearn KNN search.
    learn : fastai.learner.Learner
        The learner used for collab training
    index_samples : pd.Index, optional
        The pandas.Index for the training samples.
        If no index_samples is provided, the samples will just be numbered,
        by default None 

    Returns
    -------
    pd.DataFrame
        predictions as DataFrame for all features encoded by the model for all samples.
        
    """
    # Matrix multiplication way
    test_sample_embeddings = learn.u_weight(
        idx_samples).mean(1)  # torch.tensor
    test_sample_biases = learn.u_bias(idx_samples).mean(1)  # torch.tensor

    # first token is for NA
    idx_feat = torch.arange(1, learn.i_weight.num_embeddings)
    feat_embeddings = learn.i_weight(idx_feat)
    feat_biases = learn.i_bias(idx_feat)

    res = test_sample_embeddings.matmul(feat_embeddings.T)
    res = res + feat_biases.T + test_sample_biases

    if learn.y_range is not None:
        res = torch.sigmoid(res) * (learn.y_range[1]-learn.y_range[0]
                                    ) + learn.y_range[0]

    res = pd.DataFrame(res,
                       columns=pd.Index(list(learn.classes[learn.model.item].items[1:]),
                                        name=learn.model.item),
                       index=index_samples)
    return res


class CollabAnalysis(analysis.ModelAnalysis):

    def __init__(self,
                 datasplits: vaep.io.datasplits.DataSplits,
                 sample_column='Sample ID',
                 item_column='peptide',
                 target_column='intensity',
                 model_kwargs=dict(),
                 batch_size=64):
        if datasplits.val_y is not None:
            self.X, self.frac = combine_data(datasplits.train_X,
                                         datasplits.val_y)
        else:
            self.X, self.frac = datasplits.train_X.reset_index(), 0.0
        self.batch_size = batch_size
        self.dls = CollabDataLoaders.from_df(self.X, valid_pct=self.frac,
                                             seed=42,
                                             user_name=sample_column,
                                             item_name=item_column,
                                             rating_name=target_column,
                                             bs=self.batch_size)
        user_name=sample_column
        item_name=item_column
        rating_name=target_column
        cat_names = [user_name,item_name]
        ratings = self.X
        splits = None
        if datasplits.val_y is not None:
            idx_splitter = IndexSplitter(list(range(len(datasplits.train_X), len(datasplits.train_X)+ len(datasplits.val_y) )))
            splits = idx_splitter(self.X)
        to = TabularCollab(ratings, [Categorify], cat_names, y_names=[rating_name], y_block=TransformBlock(), splits=splits)
        self.dls = to.dataloaders(path='.', bs=self.batch_size)
        self.params = {}
        self.model_kwargs = model_kwargs
        self.params['model_kwargs'] = self.model_kwargs

        self.transform = None  # No data transformation needed
        self.learn = None


def get_missing_values(df_train_long: pd.DataFrame,
                       val_idx: pd.Index,
                       test_idx: pd.Index,
                       analysis_collab: CollabAnalysis) -> pd.Series:
    mask = df_train_long.unstack().isna().stack()
    idx_real_na = mask.loc[mask].index
    idx_real_na = (idx_real_na
                   .drop(val_idx)
                   .drop(test_idx))
    dl_real_na = analysis_collab.dls.test_dl(idx_real_na.to_frame())
    pred_real_na, _ = analysis_collab.learn.get_preds(dl=dl_real_na)
    pred_real_na = pd.Series(pred_real_na, idx_real_na, name='intensity')
    return pred_real_na
