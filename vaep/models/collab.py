import logging
from typing import Tuple, List

import pandas as pd

from fastai.collab import Module, Embedding, sigmoid_range
from fastai.collab import EmbeddingDotBias

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


def combine_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[List[int]]]:
    """Helper function to combine training and validation data in long-format. Returns
    additionally list of list of row indices for each split for further use in fastai.

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
        List of list of indices belonging to training data and list of indices belonging
        to validation data.
    """
    X = train_df.append(val_df).reset_index()

    # idx_splitter = IndexSplitter(list(range(len(data.train_X), len(data.train_X)+ len(data.val_X) )))
    # splits = idx_splitter(ana_collab.X)
    N_train, N_valid = len(train_df), len(val_df)
    splits = [list(range(0, N_train)), list(range(N_train, N_train + N_valid))]

    return X, splits
